var canvas = document.getElementById('canvas');
var clearBtn = document.getElementById('clear');
var recognizeBtn = document.getElementById('recognize');
var result = document.getElementById('result');
var ctx = canvas.getContext('2d');

function getCursorPosition(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return [x, y];
}

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, 256, 256);

var startX = null;
var startY = null;

canvas.addEventListener('mousedown', function (e) {
  var position = getCursorPosition(canvas, e);
  startX = position[0];
  startY = position[1];
});

canvas.addEventListener('mouseup', function () {
  startX = null;
  startY = null;
});

canvas.addEventListener('mouseout', function () {
  startX = null;
  startY = null;
});

clearBtn.addEventListener('click', function () {
  ctx.fillRect(0, 0, 256, 256);
  result.innerHTML = '';
});

recognizeBtn.addEventListener('click', function () {
  axios
    .post('/recognize', {
      data_url: canvas.toDataURL('image/jpg'),
    })
    .then(function (res) {
      result.innerHTML = '识别结果: ' + res.data;
    });
});

ctx.strokeStyle = 'white';
ctx.lineWidth = 10;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

canvas.addEventListener('mousemove', function (e) {
  if (startX !== null && startY !== null) {
    var position = getCursorPosition(canvas, e);
    var x = position[0];
    var y = position[1];
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(x, y);
    ctx.stroke();
    startX = x;
    startY = y;
  }
});
