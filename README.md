# Neural Photo Editor

<p align="center">
  <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/neuralPhotoEditor.png" alt="Neural Photo Editor" width="60%">
</p>

## 🔍 Overview

Neural Photo Editor is an AI-powered application that enables intuitive image editing through deep generative models. The tool allows users to modify specific facial features by simply painting over them, with the AI generating realistic and coherent changes that seamlessly integrate with the rest of the image.

## ✨ Results

<p align="center">
  <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/NeuralPhotoEditorResults.png" alt="Results" width="60%">
</p>

## 🛠️ Features

- **Intuitive Brush Interface**: Paint directly on images to edit specific areas
- **Automatic Segmentation**: Uses semantic segmentation masks for precise facial feature selection
- **Real-time Preview**: See your edits as you paint
- **Smart Region Selection**: Click on segmentation regions to edit entire facial features at once
- **Super Resolution**: Enhances low-resolution edited images to high-quality outputs
- **Adjustable Editing Parameters**: Control brush size and diffusion timestep strength

## 🚀 How to Use

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>1️⃣ Upload</h3>
      <p>Upload an image to display semantic segmentation masks.</p>
      <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/npe-1.png" alt="Upload" width="100%">
    </td>
    <td width="50%" valign="top">
      <h3>2️⃣ Select Paint Brush</h3>
      <p>Choose the paint brush tool from the toolbar to begin editing.</p>
      <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/npe-2.png" alt="Select Paint Brush" width="100%">
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>3️⃣ Draw using Segmentation</h3>
      <p>Use the segmentation masks to precisely select and modify facial features.</p>
      <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/npe-3.png" alt="Draw using Segmentation" width="100%">
    </td>
    <td width="50%" valign="top">
      <h3>4️⃣ Draw using Free Hand to modify facial features.</h3>
      <p>For more custom edits, use the free hand drawing mode.</p>
      <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/npe-4.png" alt="Draw using Free Hand" width="100%">
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>5️⃣ Edit/Enhance</h3>
      <p>Apply enhancements and edits to your selected areas.</p>
      <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/npe-5.png" alt="Edit/Enhance" width="100%">
    </td>
    <td width="50%" valign="top">
      <h3>6️⃣ Save</h3>
      <p>Save your edited image when you're satisfied with the results.</p>
      <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/npe-6.png" alt="Save" width="100%">
    </td>
  </tr>
</table>

## 🧠 Architecture

The Neural Photo Editor integrates several deep learning models working together:

### 1. Deep Diffusion Probabilistic Model (DDPM)

The primary editing model uses a diffusion-based approach to transform images.

- **Architecture**: Modified U-Net with self-attention blocks and timestep conditioning
- **Resolution**: 128x128 pixels during diffusion process
- **Training Dataset**: 224,500 human faces
- **Parameters**:
  - 500 total diffusion timestamps
  - Adam optimizer with 1e-3 learning rate
  - Linear beta scheduling (β_min=1e-4, β_max=2e-2)
  - Batch size of 40

### 2. U-Net Segmentation Model

Automatically creates semantic masks of facial features:

- **Architecture**: U-Net with 11 output channels for different facial regions
- **Dataset**: CelebAMask-HQ (30,000 high-resolution face images)
- **Classes**: 11 facial component classes including skin, eyes, nose, mouth, hair, etc.
- **Purpose**: Allows rapid selection of entire facial features without manual painting

### 3. Super-Resolution Model (GeneratorResNet)

Enhances the quality of edited images:

- **Architecture**: SR-GAN (ResNet-based generator)
- **Input Resolution**: 128x128
- **Output Resolution**: Higher quality images (up to 512x512)
- **Purpose**: Improves details in the final edited image

## 📚 The Theory Behind Diffusion Models

Diffusion models represent a powerful class of generative models that work by gradually adding noise to data and then learning to reverse this process. The fundamental principle behind diffusion models involves two key processes:

### Forward Diffusion Process

The forward process gradually adds Gaussian noise to an image across multiple timesteps (t=0 to t=T):

- At t=0, we have the original, clean image
- At each step, a small amount of noise is added according to a predefined schedule
- By t=T, the image has been transformed into pure Gaussian noise

This process is defined mathematically as:

```
q(x_t+1|x_t) = N(x_t+1; √(1-β_t+1)x_t, β_t+1I)
```

Where:
- x_t is the image at timestep t
- β_t is a scheduled noise level at timestep t
- N represents a normal distribution

### Backward Diffusion Process

The model learns to reverse the forward process, gradually removing noise to recover the original image:

- Starting from pure noise at t=T
- At each step, the model predicts the noise component of the current state
- This prediction allows reconstruction of a slightly cleaner image
- By t=0, the model has recovered a clean image

<p align="center">
  <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/ddpm.png" alt="DDPM" width="80%">
</p>

## 🔮 Harnessing Diffusion for Controlled Editing

Neural Photo Editor introduces a novel approach that leverages the power of diffusion models for targeted image editing:

### The Masked Diffusion Technique

The key innovation of our approach is the use of binary masks to control where the diffusion process applies:

1. **Partial Noise Addition**: Rather than starting from pure noise, we add noise to the user-edited image only up to a specific intermediate timestep (typically t=300-400).

2. **User Edit Preservation**: The user's painted edits define a binary mask that specifies which regions should be processed by the diffusion model.

3. **Selective Denoising**: During the reverse diffusion process, only the regions marked by the mask are denoised and transformed, while the rest of the image remains untouched.

This selective application is expressed mathematically as:

```
x_t := M*x_t + (1-M)*(√(α_t)·x_0 + √(1-α_t)·ε)
```

Where:
- M is a binary mask (1 for painted regions, 0 elsewhere)
- α_t is the cumulative product of (1-β) terms up to timestep t
- x_0 is the original image
- ε is random noise

<p align="center">
  <strong>Original Image | Painted Image | Painted Image after adding Noise | Restored/Edited Image | Controlled Restored/Edited Image</strong>
</p>

<p align="center">
  <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/neuralPhotoEditorResults1.png" alt="Neural Photo Editor" width="60%">
</p>

<p align="center">
  <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/neuralPhotoEditorResults2.png" alt="Neural Photo Editor" width="60%">
</p>

### Controlling Edit Strength with Timestep Selection

The timestep parameter offers precise control over how strongly the diffusion model transforms the edited regions:

- **Lower timesteps** (t=200-300): Subtle changes that preserve more of the original structure but still integrate the user's edits
- **Higher timesteps** (t=350-475): More dramatic transformations that fully realize the user's edits but may alter surrounding features more significantly

This timestep tuning enables a balance between edit fidelity and natural appearance.

<p align="center">
  <strong>Timestep vs Editing vs Controlled Editing</strong>
</p>

<p align="center">
  <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/editingvsT-1.png" alt="Neural Photo Editor" width="60%">
</p>

<p align="center">
  <img src="https://github.com/BhanuPrakashPebbeti/neuralPhotoEditor/blob/main/assets/editingvsT-2.png" alt="Neural Photo Editor" width="60%">
</p>

## 🔄 The Complete Editing Workflow

The Neural Photo Editor workflow seamlessly combines user interaction with advanced AI processing:

### 1. Image Preparation and Segmentation

When a user uploads an image:
- The image is automatically analyzed by the segmentation model
- A semantic mask is generated that identifies facial features (eyes, nose, lips, hair, etc.)
- This mask is displayed alongside the original image for reference

### 2. User Editing Process

The interface provides two intuitive editing modes:
- **Segmentation-based editing**: Users click on a segmentation region to apply color to an entire facial feature (e.g., changing all hair at once)
- **Free-hand painting**: Users can paint directly on the image with adjustable brush sizes for fine control

As the user paints, a binary diffusion mask is created in parallel, marking exactly which pixels should be processed by the diffusion model.

### 3. Diffusion-Based Enhancement

When the enhancement process begins:
1. **Noise Addition**: Controlled noise is added to the user-edited image up to the selected timestep
2. **Masked Denoising**: The diffusion model runs backwards from this noisy state, but only modifies the masked regions
3. **Detail Preservation**: Unmasked regions remain unchanged, ensuring that only user-selected features are modified

This approach allows for highly localized edits while maintaining global consistency and photorealism.

### 4. Super-Resolution Enhancement

Finally, the edited image is passed through the super-resolution model to enhance details and produce a polished result.

## 🏆 Why Diffusion Models Outperform Alternatives

Our extensive experimentation revealed several advantages of diffusion models over alternatives:

### Compared to Introspective Adversarial Networks (IAN)

- **Better Stability**: Diffusion models avoid the adversarial training instability of IANs
- **Superior Local Edits**: IANs struggle with localized edits, often affecting the entire image
- **Higher Fidelity**: Diffusion models preserve details better than IAN reconstructions

### Compared to Vector Quantized Variational Auto Encoders (VQVAE)

- **Smoother Results**: Diffusion avoids the blocky artifacts sometimes seen in VQVAE outputs
- **More Natural Transitions**: Diffusion creates more natural blending between edited and original regions
- **Greater Flexibility**: Timestep tuning allows more control than VQVAE's discrete latent spaces

## ⚠️ Limitations and Future Directions

While powerful, our current implementation has some limitations:

- **Resolution Constraints**: Currently limited to 128x128 resolution during diffusion
- **Domain Specificity**: Primarily optimized for facial editing
- **Computation Speed**: The full diffusion process takes several seconds

Future research directions include:

- Increasing resolution to 512x512 for the main diffusion model
- Implementing DDIM sampling for 10-20x faster inference
- Adding text-guided editing capabilities
- Extending to more image domains beyond faces
