// import * as tf from '/tf.min.js'

//Modelo
const model = tf.sequential()

model.add(tf.layers.dense({inputShape: [1], units: 1, activation: 'relu'}))
// model.add(tf.layers.dense({units: 1, activation: 'relu'}))
// model.add(tf.layers.dense({units: 1, activation: 'relu'}))

model.compile({
    loss: "meanSquaredError",
    optimizer: "sgd",
    metrics: ['accuracy']
})

const exampleY = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 1])
const exampleX = tf.tensor2d([1, 4, 9, 16, 25, 36, 49, 64, 91, 100], [10, 1])

const surface = {name: 'Historial', tab: 'Entrenamiento'}

model.fit(exampleX, exampleY, {
    epochs: 1000,
    batchSize: 32,
})
.then(log => {
    tfvis.show.history(surface, log, ['loss', 'acc'])
    tfvis.visor().open()
    
    model.predict(tf.tensor2d([2, 4, 6, 8, 10], [5, 1])).print()

})



