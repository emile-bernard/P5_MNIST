// Daniel Shiffman
// Nature of Code: Intelligence and Learning
// https://github.com/shiffman/NOC-S17-2-Intelligence-Learning

// Based on "Make Your Own Neural Network" by Tariq Rashid
// https://github.com/makeyourownneuralnetwork/

// Neural Network
let nn;

// Train and Testing Data
let training;
let testing;

// Where are we in the training and testing data
// (for animation)
let trainingIndex = 0;
let testingIndex = 0;

// How many times through all the training data
let epochs = 0;

// Network configuration
let input_nodes = 784;
let hidden_nodes = 256;
let output_nodes = 10;

// Learning rate
let learning_rate = 0.1;

// How is the network doing
let totalCorrect = 0;
let totalGuesses = 0;

// Reporting status to a paragraph
let statusP;

// This is for a user drawn image
let userPixels;
let smaller;
let ux = 16;
let uy = 100;
let uw = 140;


// Image block positions
let trainDataXPos = 216;
let trainDataYPos = 16;

let testDataXPos = 280;
let testDataYPos = 16;

let guessTextXPos = 358;
let guessTextYPos = 64;

let smallerCopyImageXPos = 280;
let smallerCopyImageYPos = 100;

let networkGuessXPos = 346;
let networkGuessYPos = 100;

let networkGuessTextXPos = 356;
let networkGuessTextYPos = 148;


// Load training and testing data
// Note this is not the full dataset
// From: https://pjreddie.com/projects/mnist-in-csv/
function preload() {
    training = loadStrings('data/mnist_train_10000.csv');
    testing = loadStrings('data/mnist_test_1000.csv');
}

function setup() {
    // Canvas
    createCanvas(520, 280);
    // pixelDensity(1);

    // Create the neural network
    nn = new NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    // Status paragraph
    statusP = createP('');
    let pauseButton = createButton('pause');
    pauseButton.mousePressed(toggle);

    // Toggle the state to start and stop
    function toggle() {
        if (pauseButton.html() == 'pause') {
            noLoop();
            pauseButton.html('continue');
        } else {
            loop();
            pauseButton.html('pause');
        }
    }

    // This button clears the user pixels
    let clearButton = createButton('clear');
    clearButton.mousePressed(clearUserPixels);
    // Just draw a black background
    function clearUserPixels() {
        userPixels.background(0);
    }

    // Save the model
    let saveButton = createButton('save model');
    saveButton.mousePressed(saveModelJSON);
    // Save all the model is a JSON file
    // TODO: add reloading functionality!
    function saveModelJSON() {
        // Take the neural network object and download
        saveJSON(nn, 'model.json');
    }


    // Create a blank user canvas
    userPixels = createGraphics(uw+30, uw+30);
    userPixels.background(0);

    // Create a smaller 28x28 image
    smaller = createImage(28, 28, RGB);
    // This is sort of silly, but I'm copying the user pixels
    // so that we see a blank image to start
    let img = userPixels.get();
    smaller.copy(img, 0, 0, uw, uw, 0, 0, smaller.width, smaller.height);
}


// When the mouse is dragged, draw onto the user pixels
function mouseDragged() {
    // Only if the user drags within the user pixels area
    if (mouseX > ux && mouseY > uy && mouseX < ux + uw && mouseY < uy + uw) {
        // Draw a white circle
        userPixels.fill(255);
        userPixels.stroke(255);
        userPixels.ellipse(mouseX - ux, mouseY - uy, 16, 16);
        // Sample down into the smaller p5.Image object
        let img = userPixels.get();
        smaller.copy(img, 0, 0, uw, uw, 0, 0, smaller.width, smaller.height);
    }
}


function draw() {
    // Train (this does just one image per cycle through draw)
    let traindata = train();

    // Test
    let result = test();
    // The results come back as an array of 3 things
    // Input data
    let testdata = result[0];
    // What was the guess?
    let guess = result[1];
    // Was it correct?
    let correct = result[2];

    // Draw the training and testing image
    drawImage(traindata, trainDataXPos, trainDataYPos, 2, 'training');

    drawImage(testdata, testDataXPos, testDataYPos, 2, 'test');

    // Draw the resulting guess
    fill(0);
    rect(346, 16, 2 * 28, 2 * 28);
    // Was it right or wrong?
    if (correct) {
        fill(0, 255, 0);
    } else {
        fill(255, 0, 0);
    }
    textSize(60);

    text(guess, guessTextXPos, guessTextYPos);

    // Tally total correct
    if (correct) {
        totalCorrect++;
    }
    totalGuesses++;

    // Show accuracy and # of epochs
    let status = 'accuracy: ' + nf(totalCorrect / totalGuesses, 0, 2);
    status += '<br>';
    // Percent correct since the sketch began
    let percent = 100 * trainingIndex / training.length;
    status += 'epochs: ' + epochs + ' (' + nf(percent, 1, 2) + '%)';
    statusP.html(status);

    // Draw the user pixels
    image(userPixels, ux, uy);
    fill(0);
    textSize(12);
    text('Draw here:', ux, 90);
    // Draw the sampled down image
    image(smaller, smallerCopyImageXPos, smallerCopyImageYPos, 28 * 2, 28 * 2);

    // Change the pixels from the user into network inputs
    let inputs = [];
    smaller.loadPixels();
    for (let i = 0; i < smaller.pixels.length; i += 4) {
        // Just using the red channel since it's a greyscale image
        // Not so great to use inputs of 0 so smallest value is 0.01
        inputs[i / 4] = map(smaller.pixels[i], 0, 255, 0, 0.99) + 0.01;
    }
    // Get the outputs
    let outputs = nn.query(inputs);
    // What is the best guess?
    guess = findMax(outputs);

    // Draw the resulting guess
    fill(0);
    rect(networkGuessXPos, networkGuessYPos, 2 * 28, 2 * 28);
    fill(255);
    textSize(60);
    text(guess, networkGuessTextXPos, networkGuessTextYPos);
}

// Function to train the network
function train() {
    // Grab a row from the CSV
    let values = training[trainingIndex].split(',');

    // Make an input array to the neural network
    let inputs = [];

    // Starts at index 1
    for (let i = 1; i < values.length; i++) {
        // Normalize the inputs 0-1, not so great to use inputs of 0 so add 0.01
        inputs[i - 1] = map(Number(values[i]), 0, 255, 0, 0.99) + 0.01;
    }

    // Now create an array of targets
    let targets = [];
    // Everything by default is wrong
    for (let k = 0; k < output_nodes; k++) {
        targets[k] = 0.01;
    }
    // The first spot is the class
    let label = Number(values[0]);
    // So it should get a 0.99 output
    targets[label] = 0.99;
    //console.log(targets);

    // Train with these inputs and targets
    nn.train(inputs, targets);

    // Go to the next training data point
    trainingIndex++;
    if (trainingIndex == training.length) {
        trainingIndex = 0;
        // Once cycle through all training data is one epoch
        epochs++;
    }

    // Return the inputs to draw them
    return inputs;
}


// Function to test the network
function test() {
    // Grab a row from the CSV
    let values = training[testingIndex].split(',');

    // Make an input array to the neural network
    let inputs = [];

    // Starts at index 1
    for (let i = 1; i < values.length; i++) {
        // Normalize the inputs 0-1, not so great to use inputs of 0 so add 0.01
        inputs[i - 1] = map(Number(values[i]), 0, 255, 0, 0.99) + 0.01;
    }

    // The first spot is the class
    let label = Number(values[0]);

    // Run the data through the network
    let outputs = nn.query(inputs);

    // Find the index with the highest probability
    let guess = findMax(outputs);

    // Was the network right or wrong?
    let correct = false;
    if (guess == label) {
        correct = true;
    }

    // Switch to a new testing data point every so often
    if (frameCount % 30 === 0) {
        testingIndex++;
        if (testingIndex == testing.length) {
            testingIndex = 0;
        }
    }

    // For reporting in draw return the results
    return [inputs, guess, correct];
}

// A function to find the maximum value in an array
function findMax(list) {
    // Highest so far?
    let record = 0;
    let index = 0;
    // Check every element
    for (let i = 0; i < list.length; i++) {
        // Higher?
        if (list[i] > record) {
            record = list[i];
            index = i;
        }
    }
    // Return index of highest
    return index;
}

// Draw the array of floats as an image
function drawImage(values, xoff, yoff, w, txt) {
    // it's a 28 x 28 image
    let dim = 28;

    // For every value
    for (let k = 0; k < values.length; k++) {
        // Scale up to 256
        let brightness = values[k] * 256;
        // Find x and y
        let x = k % dim;
        let y = floor(k / dim);
        // Draw rectangle
        fill(brightness);
        noStroke();
        rect(xoff + x * w, yoff + y * w, w, w);
    }

    // Draw a label below
    fill(0);
    textSize(12);
    text(txt, xoff, yoff + w * 35);
}