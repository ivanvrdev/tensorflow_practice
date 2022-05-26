
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

    // model.add(tf.layers.dense({inputShape: [1], units: 1}))
    model.add(tf.layers.dense({inputShape: [1], units: 20, activation: 'relu'}))
    model.add(tf.layers.dense({inputShape: [1], units: 20, activation: 'relu'}))
    model.add(tf.layers.dense({units: 1}))

    model.compile({
        loss: "meanSquaredError",
        optimizer: "adam",
        metrics: ['accuracy']
    })

    const surface = {name: 'Visor', tab: 'Entrenamiento'}

    await model.fit(tensorX, tensorY, {
        epochs: 30,
        batchSize: 64,
        callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc'])
    })

    return model
}

const loadModel = async (path) => {
    const model = await tf.loadLayersModel(path)
    return model
}

const saveModel = () =>{
    model.save('downloads://my-model').then(()=>console.log("Modelo guardado"))
}

const predictSquare = (num) =>{
    model.predict(tf.tensor([num])).print()
}

let model = undefined

//html tags
const trainButton = document.getElementById("train")
const saveButton = document.getElementById("save")
const loadButton = document.getElementById("load")
const predictButton = document.getElementById("predict")

saveButton.style.display = "none"
predictButton.style.display = "none"

//events
trainButton.addEventListener("click", async () =>{
    loadButton.style.display = "none"
    trainButton.style.display = "none"
    model = await trainModel()
    saveButton.style.display = "block"
    predictButton.style.display = "block"
})

saveButton.addEventListener("click", ()=>{
    saveModel()
})

loadButton.addEventListener("click", async ()=>{
    trainButton.style.display = "none"
    loadButton.style.display = "none"
    model = await loadModel('model/my-model.json')
    predictButton.style.display = "block"
})

predictButton.addEventListener("click", ()=>{
    predictSquare(3)
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



