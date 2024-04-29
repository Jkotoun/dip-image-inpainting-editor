# Smart Object Remover
## Easily remove objects from images
To remove an object:
- Click on an object - Mobile Segment Anything model selects it
- Refine selection by adding more selection points or using the brush tool
- Remove selected area from the image by clicking the *Remove* button - MI-GAN model fills the removed area
Available at: https://jkotoun.github.io/dip-image-inpainting-editor/
## Install dependencies:
**pnpm install**
## Run in dev environment:
**pnpm run dev**
## Or build and run production version:
**pnpm run build** && **pnpm run preview**
