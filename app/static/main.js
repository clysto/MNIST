var canvas = document.getElementById('canvas');
var clearBtn = document.getElementById('clear');
var recognizeBtn = document.getElementById('recognize');
var saveBtn = document.getElementById('save');
var result = document.getElementById('result');
var label = document.getElementById('label');
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
      var data = res.data.result.map(function (value, index) {
        return {
          number: index,
          value: value.toFixed(5),
        };
      });

      data.sort(function (a, b) {
        return b.value - a.value;
      });

      result.innerHTML = '';
      var resultTxt = document.createElement('div');
      resultTxt.className = 'console';
      var resultTable = document.createElement('table');
      var $tr = document.createElement('tr');
      $tr.innerHTML = '<th>label</th><th>value</th>';
      resultTable.appendChild($tr);
      for (var i = 0; i < 10; i++) {
        $tr = document.createElement('tr');
        $tr.innerHTML = `<td>${data[i].number}</td><td>${data[i].value}</td>`;
        resultTable.appendChild($tr);
      }
      resultTxt.innerText = res.data.number;
      result.appendChild(resultTxt);
      result.appendChild(resultTable);
    });
});

saveBtn.addEventListener('click', function () {
  var value = label.value;
  if (!value) {
    Toastify({
      text: '没有标签',
      className: 'error',
    }).showToast();
    return;
  }
  axios
    .post('/save', {
      data_url: canvas.toDataURL('image/jpg'),
      label: value,
    })
    .then(function () {
      ctx.fillRect(0, 0, 256, 256);
      result.innerHTML = '';
      Toastify({
        text: '保存成功',
        className: 'info',
      }).showToast();
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
