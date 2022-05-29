
const trainModel = async () =>{
    // datos
    let dataX = []
    let dataY = []

    const dataSize = 10
    const stepSize = 0.001

    for (let index = 0; index < dataSize; index+=stepSize) {
        dataX.push(index)
        dataY.push(index * index)
    }

    //Modelo

    const tensorX = tf.tensor(dataX)
    const tensorY = tf.tensor(dataY)

    const model = tf.sequential()

    model.add(tf.layers.dense({inputShape: [1], units: 20, activation: 'relu'}))
    model.add(tf.layers.dense({units: 20, activation: 'relu'}))
    model.add(tf.layers.dense({units: 20, activation: 'relu'}))
    model.add(tf.layers.dense({units: 1}))

    model.compile({
        loss: "meanSquaredError",
        optimizer: "adam",
        metrics: ['accuracy']
    })

    const surface = {name: 'Visor', tab: 'Entrenamiento'}

    const log = await model.fit(tensorX, tensorY, {
        epochs: 30,
        batchSize: 64,
        // callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc'])
    })
    tfvis.show.history(surface, log, ['loss', 'acc'])

    return model
}

const loadModel = async (path) => {
    const model = await tf.loadLayersModel(path)
    return model
}

const saveModel = () =>{
    model.save('downloads://model-v3').then(()=>console.log("Modelo guardado"))
}

const predictSquare = (num) =>{
    return model.predict(tf.tensor([num])).arraySync()[0][0]
}

let model = undefined

//html tags
//sections
const preTrainingButtons = document.getElementById("pre-training-buttons")
const postTrainingButtons = document.getElementById("post-training-buttons")
const loader = document.getElementById("loader")
const predictSection = document.getElementById("predict-section")

//buttons
const trainButton = document.getElementById("train")
const loadButton = document.getElementById("load")
const saveButton = document.getElementById("save")
// const predictButton = document.getElementById("predict")

//p messages
const trainingMessage = document.getElementById("training-message")
const predictMessage = document.getElementById("predict-message")

//form
const predictForm = document.forms[0]

//events
trainButton.addEventListener("click", async () =>{

    //ocultar botones de pre-entrenamiento
    preTrainingButtons.classList.add("removed")
    //mensaje "entrenando"
    trainingMessage.innerHTML = "Entrenando..."

    model = await trainModel()

    trainingMessage.innerHTML = "Modelo entrenado"

    //mostrar opciones pos-entrenamiento y formulario de predicción
    postTrainingButtons.classList.remove("removed")
    predictSection.classList.remove("removed")
})

saveButton.addEventListener("click", ()=>{
    saveModel()
    postTrainingButtons.classList.add("removed")
    trainingMessage.innerHTML = "Modelo guardado"
})

loadButton.addEventListener("click", async ()=>{

    preTrainingButtons.classList.add("removed")

    model = await loadModel('model/model-v2.json')

    trainingMessage.innerHTML = "Modelo cargado"

    predictSection.classList.remove("removed")
})

predictForm.addEventListener("submit", e => {
    e.preventDefault()

    const number = e.target.number.value
    const square =  predictSquare(parseInt(number))
    predictMessage.innerHTML = `Predicción: ${square}`
})

// datos
// let dataX = []
// let dataY = []

// const dataSize = 10
// const stepSize = 0.001

// for (let index = 0; index < dataSize; index+=stepSize) {
//     dataX.push(index)
//     dataY.push(index * index)
// }

// //Modelo

// const tensorX = tf.tensor(dataX)
// const tensorY = tf.tensor(dataY)

// const model = tf.sequential()

// // model.add(tf.layers.dense({inputShape: [1], units: 1}))
// model.add(tf.layers.dense({inputShape: [1], units: 20, activation: 'relu'}))
// model.add(tf.layers.dense({units: 1}))

// model.compile({
//     loss: "meanSquaredError",
//     optimizer: "adam",
//     metrics: ['accuracy']
// })

// const surface = {name: 'Visor', tab: 'Entrenamiento'}

// model.fit(tensorX, tensorY, {
//     epochs: 30,
//     batchSize: 64,
//     callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc'])
// })
// .then(log => {
//     // tfvis.show.history(surface, log, ['loss', 'acc'])
//     // tfvis.visor().open()
//     model.save('downloads://my-model').then(()=>console.log("Modelo guardado en descargas"))
//     model.predict(tf.tensor([7])).print()
// })



