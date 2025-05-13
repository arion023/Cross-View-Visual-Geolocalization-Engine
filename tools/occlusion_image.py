import cv2
import numpy as np
import os, csv


ROOT_DIR="D:/datasets/Warsaw/"
DIR="tmp/"
PANO_PATH="D:/datasets/Warsaw/panos/spring/Wilanow/PIXPRO/raw/001_1.JPG"
SAT_PATH="sat/waw/2022/orto/19/001.jpg"
SAT_LIST = []

# for 7360 x 3680 px pano image
# recommended tile size is 184x184px (40x20) or 368x368px (20x10) or 736x736px (10x5)
# other example sizes: 46x46px (160x80) or 92x92px (80x40) or
TILE_SIZE = (736, 736)  # (height, width)


def generate_splitfile():
    pass

def generate_black_image(shape, dir_path, pano_id):
    balck_img = np.zeros(shape, dtype=np.uint8)
    cv2.imwrite(ROOT_DIR + dir_path + pano_id + "_black.jpg", balck_img)
    print(f"Created black image: {dir_path + pano_id + '_black.jpg'}")


def generate_occlueded_panos(pano_img, dir_path, pano_id):
    # Assuming the image is a square panorama
    print(f"Generating occluded panoramas for {pano_id}...")
    height, width, _ = pano_img.shape
    grid_size = (height / TILE_SIZE[0], width / TILE_SIZE[1])
    print(f"GRID SIZE: {grid_size[0]} x {grid_size[1]}")
    print(f"TILE SIZE: {TILE_SIZE}")
    print(f"NUMBERS OF OCCLUDED IMAGES: {grid_size[0] * grid_size[1]}")

    pano_filenames = []
    pano_paths = []
    for iy, y in enumerate(range(0, height, TILE_SIZE[0])):
        for ix, x in enumerate(range(0, width, TILE_SIZE[1])):
            # Create a copy of the original image
            occluded_pano = np.copy(pano_img)

            # Set the tile area to black
            occluded_pano[y:y + TILE_SIZE[0], x:x + TILE_SIZE[1]] = 0

            # Save the occluded panorama
            occluded_pano_filename = pano_id + f"_{iy}_{ix}.jpg"
            occluded_pano_path = dir_path + occluded_pano_filename
            pano_paths.append(occluded_pano_path)
            pano_filenames.append(occluded_pano_filename)
            cv2.imwrite(ROOT_DIR + occluded_pano_path, occluded_pano)
            print(f"Created occluded panorama: {occluded_pano_path}")

    with open(ROOT_DIR + dir_path + f"split_occluded_{pano_id}_{int(grid_size[0])}x{int(grid_size[1])}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([SAT_PATH, dir_path + pano_id + "_black.jpg","empty"]) #XD
        for pano_path in pano_paths:
            writer.writerow([SAT_PATH,  pano_path,"empty"]) #XD




def main():
    splited_pano_path = PANO_PATH.split("/")
    pano_filename = splited_pano_path[-1]
    pano_id = pano_filename.split(".")[0]
    dir_path = DIR
    if DIR is None:
        print("Creating directory in standard path.")
        dir_path = "/".join(splited_pano_path[:-1]) + "/occluded/"

    pano_img = cv2.imread(PANO_PATH)
    shape = pano_img.shape
    grid_size = ( shape[0] / TILE_SIZE[0], shape[1] / TILE_SIZE[1])

    dir_path = dir_path + pano_id + f"/{int(grid_size[0])}x{int(grid_size[1])}/"
    if not os.path.exists(ROOT_DIR + dir_path):
        os.makedirs(ROOT_DIR + dir_path)
        print(f"Created directory: {ROOT_DIR + dir_path}")

    generate_black_image(shape, dir_path, pano_id)
    generate_occlueded_panos(pano_img, dir_path, pano_id)







if __name__ == "__main__":
    main()