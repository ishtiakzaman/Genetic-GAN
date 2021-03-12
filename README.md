# Genetic-GAN
Synthesizing images between two domains by genetic crossover.

Synthesizing an interpolated image between two real images can be achieved by a simple interpolation on the latent space of the images, so that the resulting image inherits features from both. The task becomes more difficult when two images are in different domains, because an interpolated image whose latent representation lies near the middle of two distant input images may not be realistic and may end up in either domain. In this paper, we present a novel technique called Genetic-GAN that solves a novel problem of synthesizing a set of images that inherit features from both of the domains, while at the same time allowing control of which domain the resulting images fall into. We experiment on human face images using female and male genders as two different domains. We show that our method can take two images with very different attributes and synthesize images between them, and can perform domain transformations.



![Gentic-GAN](/example1_2.jpg?raw=true "Result of Genetic-GAN")
![Gentic-GAN](/example6_2.jpg?raw=true "Genetic-GAN Architecture")
![Gentic-GAN](/example7_2.jpg?raw=true "Result of Genetic-GAN")
