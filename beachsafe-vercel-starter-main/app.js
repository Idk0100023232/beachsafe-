const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const resultsEl = document.getElementById('results');
let model = null;
let playing = false;
let rafId = null;

canvas.width = 640; canvas.height = 480;

async function loadModel(){
  status.textContent = 'Loading COCO-SSD model...';
  model = await cocoSsd.load();
  status.textContent = 'Model loaded — ready.';
  tick();
}

// File upload
document.getElementById('file').addEventListener('change', e => {
  const f = e.target.files[0];
  if(!f) return;
  stopWebcam();
  video.src = URL.createObjectURL(f);
  video.onloadedmetadata = () => { video.play(); playing = true; };
});

// Webcam
document.getElementById('webcamBtn').addEventListener('click', async () => {
  try{
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();
    playing = true;
  }catch(err){ alert('Webcam error: ' + err.message); }
});

document.getElementById('pauseBtn').addEventListener('click', ()=>{
  if(playing){ video.pause(); playing = false; document.getElementById('pauseBtn').textContent = 'Resume'; }
  else{ video.play(); playing = true; document.getElementById('pauseBtn').textContent = 'Pause'; }
});

function stopWebcam(){
  const s = video.srcObject;
  if(s && s.getTracks){ s.getTracks().forEach(t => t.stop()); }
  video.srcObject = null;
}

async function tick(){
  if(!model){ rafId = requestAnimationFrame(tick); return; }
  if(playing && video.readyState >= 2){
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const predictions = await model.detect(canvas);

    resultsEl.innerHTML = '';
    const showBoxes = document.getElementById('showBoxes').checked;
    predictions.forEach(p => {
      const [x,y,w,h] = p.bbox;
      if(showBoxes){
        ctx.lineWidth = 2; ctx.strokeStyle = 'red'; ctx.strokeRect(x,y,w,h);
        ctx.fillStyle = 'red'; ctx.font = '16px Arial';
        ctx.fillText(`${p.class} ${(p.score*100).toFixed(0)}%`, x+4, y+16);
      }

      const li = document.createElement('div');
      li.textContent = `${p.class} — ${(p.score*100).toFixed(0)}%`;
      resultsEl.appendChild(li);
    });
  }
  rafId = requestAnimationFrame(tick);
}

// Load model on startup
loadModel();
