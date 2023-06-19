const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4'
const EXAMPLE_IMG = document.getElementById('exampleImg')
const container = document.getElementById('container')
const reloadImgButton = document.getElementById('reloadImgButton')
reloadImgButton.addEventListener('click', () => location.reload())

let movenet = undefined
let ssdCocoModel = undefined
cocoSsd.load().then((loadedModel) => {
  ssdCocoModel = loadedModel
})

async function loadAndRunModel() {
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true })

  if (!ssdCocoModel) { return }

  let exempleInputTensor = tf.zeros([1, 192, 192, 3], 'int32')
  let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG)

  ssdCocoModel.detect(EXAMPLE_IMG).then(async (predictions) => {

    if (predictions.length == 0) { location.reload() }


    for (let i = 0; i < predictions.length; i++) {
      const prediction = predictions[i]
      const confidence = Math.round(parseFloat(prediction.score) * 100)

      if (confidence > 50 && prediction.class == 'person') {

        const p = document.createElement('p')
        p.innerText = `${prediction.class} - ${confidence}%`
        p.style = `
        color: #0066cc;
        position: absolute;
        top: 1rem;
        left: 1rem;
        opacity: .5;
        `

        const box = document.createElement('div')
        const hitBox = document.createElement('div')
        let objWidth = Math.round(prediction.bbox[2] > EXAMPLE_IMG.width ? EXAMPLE_IMG.width : prediction.bbox[2])
        let objHeight = Math.round(prediction.bbox[3] > EXAMPLE_IMG.height ? EXAMPLE_IMG.height : prediction.bbox[3])

        let left = Math.round(prediction.bbox[0] - ((objHeight - objWidth) / 2))
        let top = Math.round(prediction.bbox[1])

        box.style = `
        left: ${left}px;
        top: ${top}px;
        width: ${objHeight}px;
        height: ${objHeight}px;
        z-index: 1000;
        position: absolute;
        `

        hitBox.style = `
        left: 0px;
        top:0px;
        width: 100%;
        height: 100%;
        border: 1px solid #0066cc32;
        box-shadow: 0px 0px 10px -5px #3d3d3d;
        border-radius: 1.2rem;
        position: relative;
        z-index: 1000;
        `

        box.appendChild(p)
        box.appendChild(hitBox)
        container.appendChild(box)

        let cropStartPoint = [top, left, 0]
        let cropSize = [objHeight, objHeight, 3]
        let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize)
        let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt()

        let tensorOutput = movenet.predict(tf.expandDims(resizedTensor))
        let arrayOutput = await tensorOutput.array()

        if (arrayOutput[0][0].length < 10) { location.reload() }
        console.log(arrayOutput[0][0].length);
        for (let i = 0; i < arrayOutput[0][0].length; i++) {
          const position = arrayOutput[0][0][i]
          const point = document.createElement('div')
          const confidence = Math.round(parseFloat(position[2]) * 100)
          if (confidence > 50) {

            point.style = `
            position: absolute;
            z-index: 1000;
            background: #0066cc80;
            top: ${Math.round(position[0] * objHeight)}px;
            left: ${Math.round(position[1] * objHeight)}px;
            backdrop-filter:blur(10px);
            width: 8px;
            height: 8px;
            box-shadow: 0px 0px 3px  #1d1d1d;
            border-radius: 50px;
            `

            hitBox.appendChild(point)
          }
        }

      } else { }
    }
  })

  // console.log(imageTensor.shape)

  // let tensorOutput = movenet.predict(exempleInputTensor)
  // let arrayOutput = await tensorOutput.array()

}

loadAndRunModel()