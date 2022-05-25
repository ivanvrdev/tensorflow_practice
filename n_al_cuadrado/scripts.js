// import * as tf from '/tf.min.js'

//Modelo
const model = tf.sequential()

model.add(tf.layers.dense({inputShape: [1], units: 1, activation: 'relu'}))
// model.add(tf.layers.dense({units: 2, activation: 'relu'}))
model.add(tf.layers.dense({units: 5, activation: 'relu'}))
model.add(tf.layers.dense({units: 1, activation: 'relu' }))

model.compile({
    loss: "meanSquaredError",
    optimizer: "sgd",
    metrics: ['accuracy']
})

let dataX = []
let dataY = []

for (let index = 0; index < 10; index++) {
    dataX.push(index)
    dataY.push(index * index)
    // dataY.push(Math.pow(index, 2))
}

const tensorX = tf.tensor2d(dataX, [10, 1])
const tensorY = tf.tensor2d(dataY, [10, 1])

const surface = {name: 'Historial', tab: 'Entrenamiento'}

model.fit(tensorX, tensorY, {
    epochs: 500,
    batchSize: 32,
})
.then(log => {
    tfvis.show.history(surface, log, ['loss', 'acc'])
    tfvis.visor().open()
    
    model.predict(tf.tensor2d([2, 4, 6, 8, 10], [5, 1])).print()

})



