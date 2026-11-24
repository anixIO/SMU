const byId = id => document.getElementById(id);
const urls = byId('monitor').dataset;
const csrf = document.querySelector('[name=csrfmiddlewaretoken]').value;
const video = byId('video');
const canvas = byId('capture');
const context = canvas.getContext('2d');
const speech = window.speechSynthesis;

let stream = null;
let active = false;
let serverStarted = false;
let leaving = false;
let timer = null;
let pending = null;
let closing = null;
let audio = null;
let utterance = null;
let samples = [];
let eventCount = 0;

async function post(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'X-CSRFToken': csrf },
    credentials: 'same-origin',
    signal: AbortSignal.timeout(15000),
    body,
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error('Сервер вернул неверный ответ. Проверьте терминал.');
  }
  if (!response.ok) throw new Error(data.error || 'Не удалось выполнить запрос.');
  return data;
}

function setStatus(state, title, description) {
  byId('status').dataset.state = state;
  // Не повторяем объявления для скринридера на каждом кадре.
  if (byId('status-title').textContent !== title) byId('status-title').textContent = title;
  if (byId('status-description').textContent !== description) byId('status-description').textContent = description;
}

async function prepareAudio() {
  try {
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (!audio || audio.state === 'closed') audio = new Audio();
    await audio.resume();
  } catch {
    byId('voice-hint').textContent = 'Звуковой сигнал недоступен. Голос можно проверить отдельно.';
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
    byId('voice-hint').textContent = 'Браузер не поддерживает голос. Используется звуковой сигнал.';
    beep();
    return;
  }
  speech.cancel();
  utterance = new window.SpeechSynthesisUtterance(text);
  utterance.lang = 'ru-RU';
  const voices = speech.getVoices().filter(voice => voice.lang.toLowerCase().startsWith('ru'));
  utterance.voice = voices.find(voice => voice.localService) || voices[0] || null;
  utterance.onerror = event => {
    if (event.error === 'canceled' || event.error === 'interrupted') return;
    byId('voice-hint').textContent = 'Голос недоступен. Проверьте русский голос в системе; пока используется сигнал.';
    beep();
  };
  byId('voice-hint').textContent = '';
  speech.speak(utterance);
}

function resetValues() {
  for (const id of ['ear', 'mar', 'perclos', 'pitch', 'yaw', 'roll']) byId(id).textContent = '—';
}

function releaseCamera() {
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  byId('camera-empty').hidden = false;
  byId('camera-label').hidden = true;
  byId('toggle').textContent = 'Начать наблюдение';
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
    const x = 990 - (now - sample.time) / 30000 * 955;
    const y = 118 - Math.max(0, Math.min(1, sample.value)) * 106;
    drawing += `${connected ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)} `;
    connected = true;
  }
  byId('chart-line').setAttribute('d', drawing);
  byId('chart-empty').hidden = drawing !== '';
}

function update(metrics) {
  byId('frames').textContent = metrics.frames;
  byId('fps').textContent = metrics.fps.toFixed(1);
  byId('camera-label').textContent = metrics.detected ? 'Лицо обнаружено' : 'Лицо не обнаружено';
  if (!metrics.detected) {
    resetValues();
    setStatus('idle', 'Лицо не обнаружено', 'Посмотрите в камеру. История показателей сброшена.');
  } else {
    for (const id of ['ear', 'mar']) byId(id).textContent = metrics[id].toFixed(3);
    byId('perclos').textContent = metrics.perclos_ready ? (metrics.perclos * 100).toFixed(0) + '%' : '…';
    for (const id of ['pitch', 'yaw', 'roll']) byId(id).textContent = metrics[id].toFixed(0) + '°';
    if (metrics.fatigue) {
      setStatus('warn', 'Сделайте перерыв', metrics.reasons.join('; '));
    } else {
      setStatus('ok', 'Признаков усталости нет', metrics.perclos_ready ? 'Наблюдение продолжается.' : 'Накапливаем данные PERCLOS.');
    }
  }

  const now = performance.now();
  samples.push({ time: now, value: metrics.detected && metrics.perclos_ready ? metrics.perclos : null });
  samples = samples.filter(sample => now - sample.time <= 30000);
  drawChart();

  if (metrics.event) {
    if (!eventCount) byId('events').replaceChildren();
    const item = document.createElement('li');
    const time = document.createElement('time');
    const text = document.createElement('span');
    time.textContent = metrics.event.time;
    text.textContent = metrics.event.text;
    item.append(time, text);
    byId('events').prepend(item);
    if (byId('events').children.length > 50) byId('events').lastElementChild.remove();
    byId('event-count').textContent = '· ' + (++eventCount);
    if (byId('sound').checked) speak('Обнаружена усталость. Сделайте перерыв.');
  }
}

async function sendFrame() {
  if (!active) return;
  const scale = Math.min(640 / video.videoWidth, 480 / video.videoHeight, 1);
  const width = Math.max(1, Math.round(video.videoWidth * scale));
  const height = Math.max(1, Math.round(video.videoHeight * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  // Зеркало только в предпросмотре: детектор получает исходные координаты.
  context.drawImage(video, 0, 0, width, height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .8));
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
      console.error('Ошибка остановки сеанса:', stopError);
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
  byId('toggle').disabled = true;
  closing = (async () => {
    // Последний кадр должен завершиться до освобождения детектора на сервере.
    if (pending) await pending.catch(() => {});
    if (serverStarted) {
      await post(urls.stopUrl);
      serverStarted = false;
    }
    setStatus('idle', 'Сеанс завершён', 'Результаты сохранены в CSV.');
  })();
  try {
    await closing;
  } finally {
    closing = null;
    byId('toggle').disabled = false;
  }
}

async function start() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Камера недоступна. Откройте страницу через localhost или HTTPS.');
  setStatus('idle', 'Подключаем камеру', 'Разрешите доступ к камере в браузере.');
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    audio: false,
  });
  if (leaving) { releaseCamera(); return; }
  video.srcObject = stream;
  await video.play();
  if (leaving) { releaseCamera(); return; }
  await post(urls.startUrl);
  serverStarted = true;
  if (leaving) { await stop(); return; }
  active = true;
  samples = [];
  eventCount = 0;
  resetValues();
  byId('frames').textContent = '0';
  byId('fps').textContent = '0.0';
  byId('events').innerHTML = '<li class="muted">Предупреждений пока нет.</li>';
  byId('event-count').textContent = '· 0';
  byId('chart-line').setAttribute('d', '');
  byId('chart-empty').hidden = false;
  byId('report').setAttribute('aria-disabled', 'false');
  byId('report').removeAttribute('tabindex');
  byId('camera-empty').hidden = true;
  byId('camera-label').hidden = false;
  byId('toggle').textContent = 'Завершить';
  stream.getVideoTracks()[0].addEventListener('ended', () => {
    if (active) stop().catch(error => setStatus('warn', 'Ошибка остановки', error.message));
  });
  scheduleFrame();
}

byId('toggle').addEventListener('click', async () => {
  byId('toggle').disabled = true;
  try {
    if (active) {
      await stop();
    } else {
      // Первая фраза произносится по нажатию: браузер разрешает дальнейшие оповещения.
      if (byId('sound').checked) {
        speak('Голосовые оповещения включены.');
        await prepareAudio();
      }
      await start();
    }
  } catch (error) {
    active = false;
    releaseCamera();
    speech?.cancel();
    const cameraErrors = {
      NotAllowedError: 'Разрешите доступ к камере в настройках браузера.',
      NotFoundError: 'Камера не найдена. Подключите камеру и повторите запуск.',
      NotReadableError: 'Камера занята или недоступна. Закройте другие приложения с камерой.',
    };
    setStatus('warn', 'Не удалось продолжить', cameraErrors[error.name] || error.message);
  } finally {
    byId('toggle').disabled = false;
  }
});

byId('test-voice').addEventListener('click', () => {
  speak('Обнаружена усталость. Сделайте перерыв.');
  prepareAudio();
});
byId('sound').addEventListener('change', () => {
  if (byId('sound').checked) {
    speak('Голосовые оповещения включены.');
    prepareAudio();
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
  if (serverStarted) navigator.sendBeacon(urls.stopUrl, new FormData(byId('csrf-form')));
  if (audio) audio.close();
});
