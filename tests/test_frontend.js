// Проверка логики браузера без камеры и воспроизведения звука.
// Запуск: node tests/test_frontend.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    elements.set(id, {
      textContent: '', dataset: {}, checked: false, open: false, children: [], handlers: {},
      addEventListener(type, callback) { this.handlers[type] = callback; },
      setAttribute() {}, removeAttribute() {}, append() {},
      replaceChildren() { this.children = []; }, prepend(child) { this.children.unshift(child); },
    });
  }
  return elements.get(id);
}

const spoken = [];
const requests = [];
const track = { stop() {}, addEventListener() {} };
const stream = { getTracks: () => [track], getVideoTracks: () => [track] };
Object.assign(element('monitor').dataset, { startUrl: '/start', frameUrl: '/frame', stopUrl: '/stop' });
Object.assign(element('video'), { videoWidth: 1280, videoHeight: 720, play: async () => {} });
Object.assign(element('capture'), {
  getContext: () => ({ drawImage() {} }),
  toBlob: callback => callback(new Blob(['jpeg'], { type: 'image/jpeg' })),
});
element('sound').checked = true;
let failFrame = false;
const russianVoice = { lang: 'ru-RU', localService: true };
const sandbox = {
  document: { getElementById: element, createElement: () => element(Symbol()), querySelector: () => ({ value: 'csrf' }) },
  navigator: { mediaDevices: { getUserMedia: async () => stream } },
  window: {
    addEventListener() {},
    speechSynthesis: { cancel() {}, getVoices: () => [russianVoice], speak: message => spoken.push(message) },
    SpeechSynthesisUtterance: class { constructor(text) { this.text = text; } },
    AudioContext: class { constructor() { this.state = 'running'; } async resume() {} },
  },
  console, performance, FormData, Blob, AbortController,
  setTimeout: () => 1, clearTimeout() {},
  fetch: async (url, options) => {
    requests.push(url);
    assert.equal(options.headers['X-CSRFToken'], 'csrf');
    if (url !== '/start') assert.equal(options.headers['X-Observation'], 'test-session');
    if (url === '/frame') {
      if (failFrame) throw new Error('Connection interrupted');
      return { ok: true, json: async () => ({
        detected: true, status: 'fatigue', seconds: 2, ear: .3, mar: .1, perclos: .2, perclos_ready: true,
        pitch: 40, yaw: 0, roll: 0, fatigue: true, reasons: ['наклон головы вперёд 40°'],
        frames: 1, fps: 10, event: { time: '12:00:00', text: 'наклон головы вперёд 40°' },
      }) };
    }
    return { ok: true, json: async () => ({ active: url === '/start', observation: 'test-session', has_report: true }) };
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/monitor.js'), 'utf8'), sandbox);

async function check() {
  await element('test-voice').handlers.click();
  assert.equal(spoken.at(-1).text, 'Обнаружены признаки сонливости. Сделайте перерыв.');
  assert.equal(spoken.at(-1).voice, russianVoice);
  await element('toggle').handlers.click();
  await new Promise(setImmediate);
  assert.deepEqual(requests, ['/start', '/frame']);
  assert.equal(element('capture').width, 640);
  assert.equal(element('capture').height, 360);
  assert.equal(element('status-title').textContent, 'Сделайте перерыв');
  assert.equal(spoken.at(-1).text, 'Обнаружены признаки сонливости. Сделайте перерыв.');

  const count = spoken.length;
  element('sound').checked = false;
  element('sound').handlers.change();
  vm.runInContext('update({detected:false, frames:2, fps:10, event:{time:"12:00:04", text:"проверка"}})', sandbox);
  assert.equal(spoken.length, count);
  assert.equal(element('ear').textContent, '—');
  await element('toggle').handlers.click();
  assert.equal(requests.at(-1), '/stop');

  failFrame = true;
  await element('toggle').handlers.click();
  await new Promise(setImmediate);
  assert.equal(requests.at(-1), '/stop');
  assert.equal(element('status-title').textContent, 'Наблюдение остановлено');
  assert.equal(element('toggle').disabled, false);
  console.log('Voice, mute, head-tilt warning, frame scaling and session cleanup: OK');
}
check().catch(error => { console.error(error); process.exitCode = 1; });
