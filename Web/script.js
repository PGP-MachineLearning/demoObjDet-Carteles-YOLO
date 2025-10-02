// código javascript adaptado de https://github.com/Hyuto/yolov8-tfjs/tree/master/src

let model;

// labels a identificar
const classNames = ['2', '0', '8', '6', '4', '3', '1', '7', '9', '5']
const numClass = classNames.length;

const showDebug = 0;

// tamaño modelo
let modelWidth = 480;
let modelHeight = 480;

async function loadModel() {
  // loading model
  console.log('Loading model...');
  model = await tf.loadGraphModel(
        `https://raw.githubusercontent.com/PGP-MachineLearning/demoObjDet-Carteles-YOLO/8d478633bc6c06c658e48ca9c181afee981ee8be/Modelo/tfjs/model.json`); // load model
  console.log('Model loaded. ')
  if (showDebug==1) {
	  console.log(model)
  }
  // warming up model
  console.log('Warming up Model.');
  const dummyInput = tf.ones(model.inputs[0].shape);
  const warmupResults = model.execute(dummyInput);
  if (showDebug==1) {
	  console.log('Warming up results: ', warmupResults);
  }  
 console.log('Model ready.');
}



// Call the loadModel function when the page loads
document.addEventListener('DOMContentLoaded', loadModel);

const imageUpload = document.getElementById('imageUpload');
const uploadedImage = document.getElementById('uploaded-image');
const detectionCanvas = document.getElementById('detection-canvas');

imageUpload.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
});

uploadedImage.addEventListener('load', async () => {
  detectionCanvas.width = uploadedImage.width;
  detectionCanvas.height = uploadedImage.height;

  if (model) {
    console.log('Preprocess image...');
    let xRatio, yRatio; // ratios for boxes

    const input = tf.tidy(() => {
		// load pixels
		const img = tf.browser.fromPixels(uploadedImage);
		// padding image to square => [n, m] to [n, n], n > m
		const [h, w] = img.shape.slice(0, 2); // get source width and height
		const maxSize = Math.max(w, h); // get max size
		const imgPadded = img.pad([
			  [0, maxSize - h], // padding y [bottom only]
			  [0, maxSize - w], // padding x [right only]
			  [0, 0],
			]);
		// define ratios
		xRatio = maxSize / w; // update xRatio
		yRatio = maxSize / h; // update yRatio
		// prepare image
		return tf.image
		  .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
		  .div(255.0) // normalize
		  .expandDims(0); // add batch
    });
    if (showDebug==1) {
  	  console.log("-> input: ", input);
	  console.log("-> xRatio: ", xRatio);
  	  console.log("-> yRatio: ", yRatio);
    }    	  
    console.log('Running Model...');
    const res = model.execute(input); // inference model    
    console.log('Model complete.')
    if (showDebug==1) {
  	  console.log("-> res: ", res);
  	 }
	console.log('Postprocess model results...');
	const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
	  const boxes = tf.tidy(() => {
		const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
		const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
		const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
		const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
		return tf
		  .concat(
			[
			  x1,
			  y1,
			  tf.add(x1, w), //x2
			  tf.add(y1, h) //x2
			],
			2
		  )
		  .squeeze();
	  }); // process boxes [x1, y1, x2, y2]
	  const [scores, classes] = tf.tidy(() => {
		// class scores
		const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
		return [rawScores.max(1), rawScores.argMax(1)];
	  }); // get max scores and classes index
	  // remove overlapping
	 const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes
	 const boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
	 const scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
 	 const classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index
	if (showDebug==1) {
  	  console.log("-> Results: [classes_data, boxes_data, scores_data]");
  	  console.log(classes_data, boxes_data, scores_data);
     }
	 console.log("Rendering results... ");
 	 const ctx = detectionCanvas.getContext('2d'); 
     ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height); // Clear previous drawings 	 
     const maxRatio = Math.max(xRatio, yRatio);
	 for (let i = 0; i < scores_data.length; ++i) {	 
      	  const className = classNames[classes_data[i]]
      	  const score = scores_data[i]	 
		  let [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);
			x2 = x2 * maxRatio;
			x1 = x1 * maxRatio;
			y2 = y2 * maxRatio;
			y1 = y1 * maxRatio;
       // Draw bounding box
		  ctx.strokeStyle = '#FF0000'; // Red color
		  ctx.lineWidth = 2;
		  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
		  // Draw label
		  ctx.fillStyle = '#FF0000'; // Red color for text
		  ctx.font = '16px Arial';
		  const text = `${className}: ${score.toFixed(2)}`;
		  const textWidth = ctx.measureText(text).width;
		  const textX = x1;
		  const textY = y1 > 10 ? y1 - 5 : 10; // Position text above the box, adjust if too close to top
		  // add text
		  ctx.fillText(text, textX, textY);
  	 }
 	 console.log("Freeing memory... ");
     tf.dispose([res, transRes, boxes, scores, classes, nms]); // clear memory
	 console.log("End. ");
  } else {
    console.log('Model not loaded yet.');
  }
});


