
from segment import SegmentationHelper
from segKmens import SegKMens
import matplotlib.pyplot as plt


if __name__ == "__main__":
    seg = SegmentationHelper()
    seg.read_img("ImagemFuzzyCluster01.pgm")
    img = seg.get_img()

    SKmens = SegKMens(img)
    clusters = SKmens.clusters()
    print(clusters)

    plt.imshow(img, cmap='gray')
    plt.show()

    fig, axes = plt.subplots(1, 6, figsize=(15, 3))

    for i, ax in enumerate(axes):
        # imagem de exemplo (você colocaria sua imagem real aqui)
        img = SKmens.run(i)

        ax.imshow(img, cmap='gray')
        ax.set_title(f"Imagem {i+1}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()