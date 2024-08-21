import os, ast, cv2, warnings, pygame, litemapy, numpy as np

from sklearn.cluster import KMeans
from argparse import ArgumentParser
from zipfile import ZipFile
from sys import argv


# Const variables
APPDATA_ROAMING = os.getenv("appdata")
TEMP_FOLDER = os.path.join(os.getenv("tmp"), "PictureToMinecraft")
MINECRAFT_VERSION_FOLDER = os.path.join(APPDATA_ROAMING, ".minecraft/versions")
VERBOSE = False


# Argument parser
argument_parser = ArgumentParser()
argument_parser.add_argument("--version", required=True, help="The minecraft version that you want to use")
argument_parser.add_argument("--image", required=True, help="The file that you want to be converted")
argument_parser.add_argument("--scale-factor", type=float, required=False, default=2, help="The scale factor for the output")
argument_parser.add_argument("--to-litematica", action="store_true", required=False, help="If you want the image to be converted to a litematica file")
argument_parser.add_argument("--to-png", action="store_true", required=False, help="If you want the image to be converted to a png file")
argument_parser.add_argument("--name", default="", required=False, help="The name that the image should have. Defaults to the input image's name")
argument_parser.add_argument("--dominant-color", default=False, action="store_true", required=False, help="Use the average color of the block else it will use the average color")
argument_parser.add_argument("--out-folder", required=False, help="The output folder")
argument_parser.add_argument("--verbose", default=False, action="store_true", required=False, help="Activate verbose")


def get_closest_color(colors, color: list[int]):
    color = np.array(color)
    colors = np.array(colors)

    distances = np.sqrt(np.sum((colors - color) ** 2, axis=1))
    index_of_smallest = np.where(distances == np.amin(distances))
    smallest_distance = colors[index_of_smallest]

    return smallest_distance


def get_minecraft_block_texture(block_name: str) -> str:
    return os.path.join(textures_folder, block_name + '.png')


def get_all_textures(version: str) -> list[dict[list[int], str]]:
    global textures_folder

    # Check if the version folder exists
    version_folder = os.path.join(MINECRAFT_VERSION_FOLDER, version)
    if not os.path.exists(version_folder):
        raise RuntimeError(f"The minecraft version {version} doesn't exist or hasn't been installed.")

    # Check if the jar file exists
    version_jar_file = os.path.join(version_folder, version + ".jar")
    if not os.path.exists(version_jar_file):
        raise RuntimeError(f"There has been a problem while trying to find the minecraft jar file.")
    
    jar_archive = ZipFile(version_jar_file)

    if VERBOSE: print("Extracting the models and texture files...")
    # Extract the models and the textures from the jar file
    for file in jar_archive.namelist():
        if file.startswith("assets/minecraft/models/block") or file.startswith("assets/minecraft/textures/block"):
            jar_archive.extract(file, TEMP_FOLDER)

    models_folder = os.path.join(TEMP_FOLDER, "assets/minecraft/models/block")
    textures_folder = os.path.join(TEMP_FOLDER, "assets/minecraft/textures/block")

    full_blocks: list[str] = []
    full_blocks_cache_file = os.path.join(TEMP_FOLDER, f"{version}/{version}-FullBlocks.data")

    blacklisted_blocks = ["spawner", "structure", "copper_grate", "glass", "bedrock", "leaves"]

    if os.path.exists(full_blocks_cache_file):

        if VERBOSE: print("Getting all full blocks models from cache...")
        with open(full_blocks_cache_file, 'r') as h:
            contents = h.read().strip()
            
            full_blocks = ast.literal_eval(contents)

    else:

        os.makedirs(os.path.join(TEMP_FOLDER, f"{version}"))
        open(full_blocks_cache_file, "x")

        if VERBOSE: print("Getting all full blocks models...")
        # Iterate throight all the files under models_folder
        for file in os.listdir(models_folder):
            file_name = os.path.splitext(os.path.basename(file))[0]

            with open(os.path.join(models_folder, file), "r") as h:
                contents = h.read()

                # Check if the block is a full block
                if "cube_all" in contents and not any(blacklisted_block in file_name for blacklisted_block in blacklisted_blocks):
                    print(file_name)
                    full_blocks.append(file_name)

        with open(full_blocks_cache_file, 'w') as h:
            h.write(str(full_blocks))

    averages: dict[list[int], str] = {}
    dominants: dict[list[int], str] = {}

    if VERBOSE: print("Getting all average and dominant colors...")
    for file in os.listdir(textures_folder):
        file_name = os.path.splitext(os.path.basename(file))[0]

        if file_name in full_blocks:
            full_file_path = os.path.join(textures_folder, file)

            # Get image
            img = cv2.imread(full_file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get average color
            average = img.mean(axis=0).mean(axis=0)
            average = average.tolist()

            reshape = img.reshape((img.shape[0] * img.shape[1], 3))
            
            # Ignore warnings 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cluster = KMeans(n_clusters=2).fit(reshape)
            
            # Get dominant color
            dominant = cluster.cluster_centers_[0]
            dominant = dominant.tolist()

            averages[str(average)] = file_name
            dominants[str(dominant)] = file_name

    return [averages, dominants]


def generate_blocks_array(original_path: str, averages_and_dominants: list[dict[list[int], str]], scale_factor: int) -> list[list[str]]:
    global out_image_width
    global out_image_height
    
    img = cv2.imread(original_path)

    # 16: block texture resolution
    out_image_height = int(img.shape[1] / scale_factor / 16)
    out_image_width = int(img.shape[0] / scale_factor / 16)

    img = cv2.resize(img, (out_image_height, out_image_width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cols, rows, _ = img.shape

    colors = averages_and_dominants[1] if use_dominant else averages_and_dominants[0]
    colors_values = [ast.literal_eval(i) for i in colors.keys()]

    blocks_names: list[list[str]] = np.empty((cols, rows), dtype=object)

    if VERBOSE: print("Filling block arrays...")
    for col in range(cols):
        for row in range(rows):
            pixel = img[col, row]
            pixel.tolist()

            closest_color = get_closest_color(colors_values, pixel).tolist()[0]

            block_name = list(colors.values())[colors_values.index(closest_color)]

            blocks_names[col, row] = block_name

    return blocks_names


def generate_image(blocks: list[list[str]]) -> None:
    # Pygame window
    pygame.init()
    display = pygame.display.set_mode((out_image_height * 16, out_image_width * 16))
    pygame.display.set_caption("Image To Minecraft Blocks")

    if VERBOSE: print("Generating image...")
    for col_idx, _ in enumerate(blocks):
        for row_idx, block_name in enumerate(blocks[col_idx]):
            block_texture_path = get_minecraft_block_texture(block_name)

            # Create a pygame image
            pygame_img = pygame.image.load(block_texture_path).convert()
            # Add it to the screen
            display.blit(pygame_img, (row_idx * 16, col_idx * 16), (0, 0, 16, 16))
            # Render the screen
            pygame.display.flip()
    
    # Save the screen
    pygame.image.save(display, os.path.join(out_folder, f"./{to_file_name}-Minecraft.png"))


def generate_litematica(blocks: list[list[str]]) -> None:
    reg = litemapy.Region(0, 0, 0, out_image_height, -out_image_width, 1)
    schem = reg.as_schematic(name=to_file_name, author="Picture To Litematica", description="Made with litemapy")

    if VERBOSE: print("Generating litematica...")
    for col_idx, _ in enumerate(blocks):
        for row_idx, block_name in enumerate(blocks[col_idx]):
            # Create a block
            block = litemapy.BlockState(f"minecraft:{block_name}")
            # Add it to the liteamtic region
            reg[row_idx, -col_idx, 0] = block
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Save the schematic
    schem.save(os.path.join(out_folder, f"{to_file_name}.litematic"))


def main() -> None:
    global VERBOSE
    global use_dominant
    global to_file_name
    global out_folder

    arguments = argv

    # Get all parsed arguments
    parsed_arguments = argument_parser.parse_args(arguments[1:])
    minecraft_version = parsed_arguments.version
    image_path = parsed_arguments.image
    scale_factor = parsed_arguments.scale_factor
    to_litematica = parsed_arguments.to_litematica
    to_png = parsed_arguments.to_png
    to_file_name = parsed_arguments.name
    use_dominant = parsed_arguments.dominant_color
    out_folder = parsed_arguments.out_folder
    verbose = parsed_arguments.verbose

    if verbose: VERBOSE = True

    if not os.path.exists(image_path):
        raise RuntimeError(f"The image path ({image_path}) is incorrect.")

    # to_file_name defaults to the input file name    
    if to_file_name == "":
        to_file_name = os.path.splitext(os.path.basename(image_path))[0]

    averages_and_dominants = get_all_textures(minecraft_version)
    blocks = generate_blocks_array(image_path, averages_and_dominants, scale_factor)

    if to_litematica:
        generate_litematica(blocks)
    
    if to_png:
        generate_image(blocks)

    # Delete the temp assets folder
    os.system(f"rmdir {TEMP_FOLDER}\\assets /S /Q")


if __name__ == "__main__":
    main()
