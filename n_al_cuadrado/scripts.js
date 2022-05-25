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
model.add(tf.layers.dense({units: 1}))

model.compile({
    loss: "meanSquaredError",
    optimizer: "adam",
    metrics: ['accuracy']
})

const surface = {name: 'Visor', tab: 'Entrenamiento'}

model.fit(tensorX, tensorY, {
    epochs: 30,
    batchSize: 64,
    callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc'])
})
.then(log => {
    // tfvis.show.history(surface, log, ['loss', 'acc'])
    // tfvis.visor().open()
    
    model.predict(tf.tensor([7])).print()
})



