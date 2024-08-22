import os, ast, cv2, warnings, litemapy, pygame, sys, time, numpy as np

from customtkinter import CTkLabel, CTkButton, CTkEntry, CTkCheckBox, CTkFont, CTk
from tkinter import StringVar
from tkinter.filedialog import askopenfilename, askdirectory
from sklearn.cluster import KMeans
from typing import Tuple
from argparse import ArgumentParser
from zipfile import ZipFile
from sys import argv


# Ctk Window
class Window(CTk):
    def __init__(self, fg_color: str | Tuple[str, str] | None = None, **kwargs):
        super().__init__(fg_color, **kwargs)

        self.image_file_path = ""
        self.output_folder_path = ""
        self.minecraft_version = ""
        self.to_minecraft_blocks = ""
        self.to_litematica = ""
        self.scale_factor = 1
        self.use_dominant = False

        self.title("Image to Litematica")
        self.geometry("400x500")
        self.minsize(400, 500)

        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), weight=1)
        self.grid_columnconfigure((0), weight=1)

        self.image_file_path_strvar = StringVar(master=self, value="Image: None selected.")
        self.out_folder_path_strvar = StringVar(master=self, value="Output folder: None selected.")
        self.to_litematica_check_var = StringVar(master=self, value="on")
        self.to_minecraft_blocks_check_var = StringVar(master=self, value="off")
        self.use_dominant_checkbox_var = StringVar(master=self, value="off")

        self.title_text = CTkLabel(master=self, text="Picture to Minecraft", font=CTkFont(size=26, weight="bold"))
        self.title_text.grid(column=0, row=0)
        self.title_text.grid_rowconfigure(5, weight=1)
        self.title_text.grid_columnconfigure(5, weight=1)

        self.image_file_path_text = CTkLabel(master=self, textvariable=self.image_file_path_strvar)
        self.image_file_path_text.grid(column=0, row=1)
        self.image_file_path_text.grid_rowconfigure(5, weight=1)
        self.image_file_path_text.grid_columnconfigure(5, weight=1)

        self.choose_image_btn = CTkButton(master=self, text='Select Image', command=self.ask_choose_image_file)
        self.choose_image_btn.grid(column=0, row=2)
        self.choose_image_btn.grid_rowconfigure(5, weight=1)
        self.choose_image_btn.grid_columnconfigure(5, weight=1)

        self.out_folder_path_text = CTkLabel(master=self, textvariable=self.out_folder_path_strvar)
        self.out_folder_path_text.grid(column=0, row=3)
        self.out_folder_path_text.grid_rowconfigure(5, weight=1)
        self.out_folder_path_text.grid_columnconfigure(5, weight=1)

        self.choose_out_folder = CTkButton(master=self, text='Select Output Folder', command=self.ask_choose_output_folder)
        self.choose_out_folder.grid(column=0, row=4)
        self.choose_out_folder.grid_rowconfigure(5, weight=1)
        self.choose_out_folder.grid_columnconfigure(5, weight=1)

        self.minecraft_version_entry = CTkEntry(master=self, placeholder_text="Minecraft Version...")
        self.minecraft_version_entry.grid(column=0, row=5)
        self.minecraft_version_entry.grid_rowconfigure(5, weight=1)
        self.minecraft_version_entry.grid_columnconfigure(5, weight=1)

        self.to_litematica_checkbox = CTkCheckBox(master=self, text="To Litematica", variable=self.to_litematica_check_var, onvalue="on", offvalue="off")
        self.to_litematica_checkbox.grid(column=0, row=6)
        self.to_litematica_checkbox.grid_rowconfigure(5, weight=1)
        self.to_litematica_checkbox.grid_columnconfigure(5, weight=1)

        self.to_minecraft_blocks_checkbox = CTkCheckBox(master=self, text="To Minecraft Blocks", variable=self.to_minecraft_blocks_check_var, onvalue="on", offvalue="off")
        self.to_minecraft_blocks_checkbox.grid(column=0, row=7)
        self.to_minecraft_blocks_checkbox.grid_rowconfigure(5, weight=1)
        self.to_minecraft_blocks_checkbox.grid_columnconfigure(5, weight=1)

        self.scale_factor_entry = CTkEntry(master=self, placeholder_text="Scale Factor...")
        self.scale_factor_entry.grid(column=0, row=8)
        self.scale_factor_entry.grid_rowconfigure(5, weight=1)
        self.scale_factor_entry.grid_columnconfigure(5, weight=1)

        self.use_dominant_checkbox = CTkCheckBox(master=self, text="Use Dominant", variable=self.use_dominant_checkbox_var, onvalue="on", offvalue="off")
        self.use_dominant_checkbox.grid(column=0, row=9)
        self.use_dominant_checkbox.grid_rowconfigure(5, weight=1)
        self.use_dominant_checkbox.grid_columnconfigure(5, weight=1)

        self.convert_button = CTkButton(master=self, text='Generate', command=self.generate)
        self.convert_button.grid(column=0, row=10)
        self.convert_button.grid_rowconfigure(5, weight=1)
        self.convert_button.grid_columnconfigure(5, weight=1)


    def generate(self):
        global FINISHED

        self.to_litematica = True if self.to_litematica_check_var.get() == "on" else False
        self.to_minecraft_blocks = True if self.to_minecraft_blocks_check_var.get() == "on" else False
        self.use_dominant = True if self.use_dominant_checkbox_var.get() == "on" else False
        self.minecraft_version = self.minecraft_version_entry.get()
        self.scale_factor = self.scale_factor_entry.get()

        if self.scale_factor == "":
            self.scale_factor == DEFAULT_SCALE_FACTOR

        if not is_float(self.scale_factor):
            print("Scale factor is invalid!")
            return

        self.scale_factor = float(self.scale_factor)
        
        if not self.to_litematica and not self.to_minecraft_blocks:
            print("Both To Litematia and To Minecraft bloks cannot be off at the same time!")
            return
        
        if self.minecraft_version == "":
            print("Minecraft version is invalid")
            return
     
        run(self.image_file_path, self.output_folder_path, self.minecraft_version, self.to_litematica, self.to_minecraft_blocks, self.scale_factor, self.use_dominant)


    def ask_choose_image_file(self):
        file_path = askopenfilename(filetypes=[('PNG Files', '*.png')])

        self.image_file_path = file_path
        self.image_file_path_strvar.set("Image: " + os.path.basename(file_path))


    def ask_choose_output_folder(self):
        folder_path = askdirectory()

        self.output_folder_path = folder_path
        self.out_folder_path_strvar.set("Output Folder: " + os.path.basename(folder_path))



# Const variables
APPDATA_ROAMING = os.getenv("appdata")
TEMP_FOLDER = os.path.join(os.getenv("tmp"), "PictureToMinecraft")
MINECRAFT_VERSION_FOLDER = os.path.join(APPDATA_ROAMING, ".minecraft/versions")
VERBOSE = False
DEFAULT_SCALE_FACTOR = 2.0


# Argument parser
argument_parser = ArgumentParser()
argument_parser.add_argument("--version", required=True, help="The minecraft version that you want to use")
argument_parser.add_argument("--image", required=True, help="The file that you want to be converted")
argument_parser.add_argument("--scale-factor", type=float, required=False, default=2, help="The scale factor for the output")
argument_parser.add_argument("--to-litematica", action="store_true", required=False, help="If you want the image to be converted to a litematica file")
argument_parser.add_argument("--to-png", action="store_true", required=False, help="If you want the image to be converted to a png file")
argument_parser.add_argument("--dominant-color", default=False, action="store_true", required=False, help="Use the average color of the block else it will use the average color")
argument_parser.add_argument("--out-folder", required=False, help="The output folder")
argument_parser.add_argument("--verbose", default=False, action="store_true", required=False, help="Activate verbose")


def print_progressbar(iteration: int, total: int, prefix="Progress", suffix="Complete", decimals=2, fill="█", end_line='\r'):
    percent = f"{round(100 * (iteration / float(total)), decimals)}"
    filledLength = int(100 * iteration / total)
    bar = (fill * filledLength) + ("-" * (100 - filledLength))

    print(f'{prefix} |{bar}| {percent}% {suffix}', end=end_line)

    if iteration == total: 
        print("\r")


def is_float(s):
    try: 
        float(s)
    except ValueError:
        return False
    else:
        return True
    

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

        model_files_num = len(os.listdir(models_folder))

        os.makedirs(os.path.join(TEMP_FOLDER, f"{version}"))
        open(full_blocks_cache_file, "x")

        if VERBOSE: print("Getting all full blocks models...")
        # Iterate throight all the files under models_folder
        for i, file in enumerate(os.listdir(models_folder)):
            file_name = os.path.splitext(os.path.basename(file))[0]

            with open(os.path.join(models_folder, file), "r") as h:
                contents = h.read()

                # Check if the block is a full block
                if "cube_all" in contents and not any(blacklisted_block in file_name for blacklisted_block in blacklisted_blocks):
                    full_blocks.append(file_name)

            print_progressbar(i + 1, model_files_num)

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


def generate(original_path: str, averages_and_dominants: list[dict[list[int], str]], scale_factor: int) -> None:
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
    blocks_shape = np.shape(blocks_names)
    num_blocks = blocks_shape[0] * blocks_shape[1]

    if to_litematica: 
        reg = litemapy.Region(0, 0, 0, out_image_height, -out_image_width, 1)
        schem = reg.as_schematic(name=to_file_name, author="Picture To Litematica", description="Made with litemapy")

        if VERBOSE: print("Generating litematica...")


    if to_png: 
        # Pygame window
        pygame.init()

        try: 
            window_res = (out_image_height * 16, out_image_width * 16)
            screen_res_info = pygame.display.Info()

            # If the window size is bigger than the screen's resolution, hide the window
            if window_res[0] > screen_res_info.current_w or window_res[1] > screen_res_info.current_h:
                display = pygame.display.set_mode(window_res, flags=pygame.HIDDEN)
            else:
                display = pygame.display.set_mode(window_res)
                pygame.display.set_caption("Image To Minecraft Blocks")

        except pygame.error: 
            print("Image resolution is too high. Try increasing the scale factor.")
            return
        

        if VERBOSE: print("Generating image...")

    i = 0

    for col in range(cols):
        for row in range(rows):
            pixel = img[col, row]
            pixel.tolist()

            closest_color = get_closest_color(colors_values, pixel).tolist()[0]

            block_name = list(colors.values())[colors_values.index(closest_color)]

            block_texture_path = get_minecraft_block_texture(block_name)

            if to_png:
                # Create a pygame image
                pygame_img = pygame.image.load(block_texture_path).convert()
                # Add it to the screen
                display.blit(pygame_img, (row * 16, col * 16), (0, 0, 16, 16))
                # Render the screen
                pygame.display.flip()


            if to_litematica:
                # Create a block
                block = litemapy.BlockState(f"minecraft:{block_name}")
                # Add it to the liteamtic region
                reg[row, -col, 0] = block

            i += 1

            print_progressbar(i, num_blocks)

    if to_png:
        # Save the screen
        pygame.image.save(display, os.path.join(output_folder, f"./{to_file_name}-Minecraft.png"))
        if VERBOSE: print("Finished generating image...")
        
        time.sleep(3)

        pygame.display.quit()

    if to_litematica:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the schematic
        schem.save(os.path.join(output_folder, f"{to_file_name}.litematic"))

        if VERBOSE: print("Finished generating litematica...")


def run(image_file_path: str, out_folder: str, minecraft_version: str, litematica: bool, png: bool, scale_factor: float, dominant: bool):
    global to_file_name
    global use_dominant
    global output_folder
    global to_litematica
    global to_png

    use_dominant = dominant
    output_folder = out_folder
    to_litematica = litematica
    to_png = png

    to_file_name = os.path.splitext(os.path.basename(image_file_path))[0]

    if not os.path.exists(image_file_path):
        raise RuntimeError(f"The image path ({image_file_path}) is incorrect.")

    averages_and_dominants = get_all_textures(minecraft_version)
    generate(image_file_path, averages_and_dominants, scale_factor)

    # Delete the temp assets folder
    os.system(f"rmdir {TEMP_FOLDER}\\assets /S /Q")


def main() -> None:
    global VERBOSE
    global use_dominant
    global out_folder
    global to_litematica
    global to_png
    
    arguments = argv

    if len(arguments) > 1:
        # Get all parsed arguments
        parsed_arguments = argument_parser.parse_args(arguments[1:])
        minecraft_version = parsed_arguments.version
        image_path = parsed_arguments.image
        scale_factor = parsed_arguments.scale_factor
        to_litematica = parsed_arguments.to_litematica
        to_png = parsed_arguments.to_png
        use_dominant = parsed_arguments.dominant_color
        out_folder = parsed_arguments.out_folder
        verbose = parsed_arguments.verbose

        if verbose: VERBOSE = True

        run(image_path, out_folder, minecraft_version, to_litematica, to_png, scale_factor, use_dominant)

        sys.exit(0)
    
    else:
        VERBOSE = True

        window = Window()
        window.mainloop()


if __name__ == "__main__":
    main()
