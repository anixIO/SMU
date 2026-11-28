const byId = id => document.getElementById(id);
const urls = byId('monitor').dataset;
const csrfForm = byId('csrf-form');
const csrf = document.querySelector('[name=csrfmiddlewaretoken]').value;
const video = byId('video');
const canvas = byId('capture');
const context = canvas.getContext('2d');
const toggle = byId('toggle');
const speech = window.speechSynthesis;
const perclosLabel = byId('perclos');
const cameraLabel = byId('camera-label');
const calibration = byId('calibration');

let stream = null;
let observation = null;
let active = false;
let leaving = false;
let timer = null;
let pending = null;
let closing = null;
let audio = null;
let samples = [];
let warnings = 0;
let lastMetrics = null;

async function post(url, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  const headers = { 'X-CSRFToken': csrf };
  if (observation) headers['X-Observation'] = observation;

  try {
    const response = await fetch(url, {
      method: 'POST', headers, credentials: 'same-origin', signal: controller.signal, body,
    });
    if (response.status === 403) throw new Error('Обновите страницу, чтобы восстановить доступ к сеансу.');
    const data = await response.json().catch(() => {
      throw new Error('Сервер не вернул результат. Проверьте соединение и запуск СМУ.');
    });
    if (!response.ok) throw new Error(data.error || 'Не удалось выполнить запрос.');
    return data;
  } catch (error) {
    if (error.name === 'AbortError' || error instanceof TypeError) {
      throw new Error('Нет ответа от сервера. Проверьте соединение.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function setStatus(state, title, description) {
  byId('status').dataset.state = state;
  // Скринридер не должен повторять один и тот же статус при каждом кадре.
  if (byId('status-title').textContent !== title) byId('status-title').textContent = title;
  if (byId('status-description').textContent !== description) byId('status-description').textContent = description;
}

async function prepareAudio() {
  try {
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (!audio || audio.state === 'closed') audio = new Audio();
    await audio.resume();
  } catch {
    byId('voice-hint').textContent = 'Звуковой сигнал недоступен. Можно проверить голос отдельно.';
  }
}

function beep() {
  if (!audio || audio.state !== 'running') return;
  const tone = audio.createOscillator();
  const gain = audio.createGain();
  tone.frequency.value = 660;
  tone.connect(gain);
  gain.connect(audio.destination);
  gain.gain.setValueAtTime(.12, audio.currentTime);
  gain.gain.exponentialRampToValueAtTime(.001, audio.currentTime + .5);
  tone.start();
  tone.stop(audio.currentTime + .5);
}

function speak(text) {
  if (!speech || !window.SpeechSynthesisUtterance) {
    byId('voice-hint').textContent = 'Голос недоступен. Используется звуковой сигнал.';
    beep();
    return;
  }
  try {
    speech.cancel();
    const message = new window.SpeechSynthesisUtterance(text);
    message.lang = 'ru-RU';
    const voices = speech.getVoices().filter(voice => voice.lang.toLowerCase().startsWith('ru'));
    message.voice = voices.find(voice => voice.localService) || voices[0] || null;
    message.onerror = event => {
      if (event.error === 'canceled' || event.error === 'interrupted') return;
      byId('voice-hint').textContent = 'Голос недоступен. Используется звуковой сигнал.';
      beep();
    };
    byId('voice-hint').textContent = '';
    speech.speak(message);
  } catch {
    byId('voice-hint').textContent = 'Голос недоступен. Используется звуковой сигнал.';
    beep();
  }
}

function setReport(available) {
  const report = byId('report');
  report.setAttribute('aria-disabled', String(!available));
  if (available) report.removeAttribute('tabindex');
  else report.setAttribute('tabindex', '-1');
}

function releaseCamera() {
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  byId('camera-empty').hidden = false;
  byId('camera-text').textContent = 'Выключена';
  cameraLabel.dataset.live = 'false';
  calibration.hidden = true;
  toggle.textContent = 'Начать наблюдение';
}

function drawChart() {
  if (!byId('history').open) return;
  const now = performance.now();
  let drawing = '';
  let connected = false;
  for (const sample of samples) {
    if (sample.value === null || now - sample.time > 30000) {
      connected = false;
      continue;
    }
    const x = 986 - (now - sample.time) / 30000 * 950;
    const y = 118 - Math.max(0, Math.min(1, sample.value)) * 106;
    drawing += `${connected ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)} `;
    connected = true;
  }
  byId('chart-line').setAttribute('d', drawing);
  byId('chart-empty').hidden = drawing !== '';
}

function update(metrics) {
  lastMetrics = metrics;
  byId('frames').textContent = metrics.frames;
  byId('fps').textContent = metrics.fps.toFixed(1).replace('.', ',');
  byId('camera-text').textContent = metrics.detected ? 'Лицо в кадре' : 'Лицо вне кадра';
  byId('ear').textContent = metrics.detected ? metrics.ear.toFixed(3).replace('.', ',') : '—';
  byId('mar').textContent = metrics.detected ? metrics.mar.toFixed(3).replace('.', ',') : '—';
  perclosLabel.textContent = metrics.perclos_ready ? (metrics.perclos * 100).toFixed(0) + '%' : '—';
  for (const id of ['pitch', 'yaw', 'roll']) {
    byId(id).textContent = Number.isFinite(metrics[id]) ? metrics[id].toFixed(0) + '°' : '—';
  }
  calibration.hidden = metrics.status !== 'calibrating';
  if (!calibration.hidden) calibration.value = metrics.calibration || 0;

  if (!metrics.detected) {
    setStatus('idle', 'Лицо вне кадра', 'Сядьте ближе и направьте свет на лицо.');
  } else if (metrics.status === 'calibrating') {
    setStatus('idle', 'Калибровка', 'Смотрите прямо в камеру. Можно моргать.');
  } else if (metrics.fatigue) {
    setStatus('warn', 'Сделайте перерыв', metrics.reasons.join(' · '));
  } else if (metrics.status === 'attention') {
    setStatus('attention', 'Обратите внимание', metrics.reasons.join(' · '));
  } else {
    setStatus('ok', 'Признаков сонливости нет', metrics.perclos_ready ?
      'Наблюдение продолжается.' : 'Накапливаем данные о закрытии глаз.');
  }

  const now = performance.now();
  samples.push({ time: now, value: metrics.perclos_ready ? metrics.perclos : null });
  samples = samples.filter(sample => now - sample.time <= 30000);
  drawChart();

  if (metrics.event) {
    const list = byId('events');
    if (!warnings) list.replaceChildren();
    const item = document.createElement('li');
    const time = document.createElement('time');
    const text = document.createElement('span');
    time.textContent = metrics.event.time;
    text.textContent = metrics.event.text;
    item.append(time, text);
    list.prepend(item);
    if (list.children.length > 50) list.lastElementChild.remove();
    byId('event-count').textContent = String(++warnings);
    if (byId('sound').checked) speak('Обнаружены признаки сонливости. Сделайте перерыв.');
  }
}

async function sendFrame() {
  if (!active || !video.videoWidth || !video.videoHeight) return;
  const scale = Math.min(640 / video.videoWidth, 480 / video.videoHeight, 1);
  const width = Math.max(1, Math.round(video.videoWidth * scale));
  const height = Math.max(1, Math.round(video.videoHeight * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.drawImage(video, 0, 0, width, height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .78));
  if (!active) return;
  if (!blob) throw new Error('Не удалось получить кадр камеры.');
  const form = new FormData();
  form.append('frame', blob, 'frame.jpg');
  const metrics = await post(urls.frameUrl, form);
  if (active) update(metrics);
}

function scheduleFrame() {
  pending = sendFrame();
  pending.then(() => {
    if (active) timer = setTimeout(scheduleFrame, 80);
  }).catch(async error => {
    if (!active) return;
    try {
      await stop();
    } catch (stopError) {
      console.error('Ошибка завершения сеанса:', stopError);
    }
    setStatus('warn', 'Наблюдение остановлено', error.message);
  });
}

async function stop() {
  if (closing) return closing;
  active = false;
  clearTimeout(timer);
  releaseCamera();
  speech?.cancel();
  toggle.disabled = true;
  setStatus('idle', 'Завершение сеанса', 'Сохраняем показатели.');
  closing = (async () => {
    if (pending) await pending.catch(() => {});
    if (observation) {
      const result = await post(urls.stopUrl);
      setReport(result.has_report);
      observation = null;
    }
    const info = lastMetrics?.frames
      ? `${lastMetrics.seconds} с наблюдения · ${warnings} предупреждений`
      : 'Сеанс закрыт.';
    setStatus('idle', 'Сеанс завершён', info);
  })();
  try {
    await closing;
  } finally {
    closing = null;
    toggle.disabled = false;
  }
}

async function start() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Откройте СМУ по HTTPS или через localhost.');
  setStatus('idle', 'Подключаем камеру', 'Разрешите доступ к камере в браузере.');
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    audio: false,
  });
  if (leaving) { releaseCamera(); return; }
  video.muted = true;
  video.srcObject = stream;
  let playbackTimeout;
  try {
    await Promise.race([
      video.play(),
      new Promise((_, reject) => {
        playbackTimeout = setTimeout(() => reject(new Error('Камера не передаёт изображение. Попробуйте снова.')), 8000);
      }),
    ]);
  } finally {
    clearTimeout(playbackTimeout);
  }
  if (leaving) { releaseCamera(); return; }
  byId('camera-empty').hidden = true;
  byId('camera-text').textContent = 'Подключена';
  cameraLabel.dataset.live = 'true';
  setStatus('idle', 'Подключаем анализ', 'Ожидаем ответ сервера.');
  const result = await post(urls.startUrl);
  observation = result.observation;
  if (leaving) { await stop(); return; }
  active = true;
  lastMetrics = null;
  samples = [];
  warnings = 0;
  for (const id of ['ear', 'mar', 'perclos', 'pitch', 'yaw', 'roll']) byId(id).textContent = '—';
  byId('frames').textContent = '0';
  byId('fps').textContent = '0,0';
  byId('events').innerHTML = '<li class="muted">Предупреждений пока нет.</li>';
  byId('event-count').textContent = '';
  byId('chart-line').setAttribute('d', '');
  byId('chart-empty').hidden = false;
  setReport(false);
  toggle.textContent = 'Завершить';
  stream.getVideoTracks()[0].addEventListener('ended', () => {
    if (active) stop().catch(error => setStatus('warn', 'Ошибка завершения', error.message));
  });
  scheduleFrame();
}

toggle.addEventListener('click', async () => {
  toggle.disabled = true;
  try {
    if (active) {
      await stop();
    } else {
      if (byId('sound').checked) {
        prepareAudio();
        speak('Голосовые оповещения включены.');
      }
      await start();
    }
  } catch (error) {
    active = false;
    releaseCamera();
    speech?.cancel();
    const cameraErrors = {
      NotAllowedError: 'Разрешите доступ к камере в настройках сайта.',
      NotFoundError: 'Камера не найдена.',
      NotReadableError: 'Камера занята или недоступна.',
    };
    setStatus('warn', 'Не удалось продолжить', cameraErrors[error.name] || error.message);
  } finally {
    toggle.disabled = false;
  }
});

byId('test-voice').addEventListener('click', () => {
  prepareAudio();
  speak('Обнаружены признаки сонливости. Сделайте перерыв.');
});
byId('sound').addEventListener('change', () => {
  if (byId('sound').checked) {
    prepareAudio();
    speak('Голосовые оповещения включены.');
  } else {
    speech?.cancel();
  }
});
byId('history').addEventListener('toggle', drawChart);
window.addEventListener('pageshow', () => { leaving = false; });
window.addEventListener('pagehide', () => {
  leaving = true;
  active = false;
  clearTimeout(timer);
  releaseCamera();
  speech?.cancel();
  if (observation && navigator.sendBeacon) {
    const form = new FormData(csrfForm);
    form.append('observation', observation);
    navigator.sendBeacon(urls.stopUrl, form);
  }
  if (audio && audio.state !== 'closed') audio.close();
});
