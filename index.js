const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const nsfw = require("nsfwjs");

const app = express();

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*'); // ou especifique o domínio do seu aplicativo React Native
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

const upload = multer({
  limits: {
    fileSize: 5 * 1024 * 1024, // Limite de 5MB para o tamanho da imagem
  },
});

// Carregar o modelo NSFW
let model;

const init = async () => {
  const shape_size = 224;

  try {
    model = await nsfw.load(undefined, {
      size: parseInt(shape_size),
    });
    console.info("The NSFW Model was loaded successfully!");
  } catch (err) {
    console.error(err);
  }
};

init();

// Rota para lidar com o upload da imagem
app.post("/classify", upload.single("image"), async (req, res) => {
  try {
    if (!model) {
      throw new Error("Model is not loaded yet!");
    }

    const image = await tf.node.decodeImage(req.file.buffer, 3);
    const predictions = await model.classify(image);

    console.log(predictions);
    
    image.dispose();

    res.json(predictions);
  } catch (err) {
    console.error("Error processing image:", err);
    res.status(500).json({ error: "Error processing image" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
