import numpy as np
from termcolor import colored

# (0, 0, 0) - Black
# (127, 127, 127) - Gray
# (136, 0, 21) - Bordeaux
# (237, 28, 36) - red
# (255, 127, 39) - orange
# (255, 242, 0) - yellow
# (34, 177, 76) - green
# (203, 228, 253) - blue
# (0, 162, 232) - dark blue
# (63, 72, 204) - purple
# (255, 255, 255) - white
# (195, 195, 195) - light gray
# (185, 122, 87) - light brown
# (255, 174, 201) - light pink
# (255, 201, 14) - dark yellow
# (239, 228, 176) - light yellow
# (181, 230, 29) - light green
# (153, 217, 234) - light blue
# (112, 146, 190) - dark blue
# (200, 191, 231) - light purple

colors_dict = {
    (181, 230, 99): 'green',
    (203, 228, 253): 'blue',
    (63, 72, 204): 'magenta',
    (237, 28, 36): 'red'
}

list_of_colors = [(181, 230, 99), (203, 228, 253), (63, 72, 204), (237, 28, 36)]
color1 = [12, 34, 156]
color2 = (240, 25, 40)
color3 = 189, 240, 100


def closest(colors_list, color):
    color = np.array(color)
    index = np.sqrt(np.sum((colors_list - color) ** 2, axis=1)).argmin()
    smallest_distance = colors_list[index]
    return smallest_distance


print(colored(closest(list_of_colors, color1), colors_dict[(closest(list_of_colors, color1))]))
print(colored(closest(list_of_colors, color2), colors_dict[(closest(list_of_colors, color2))]))
print(colored(closest(list_of_colors, color3), colors_dict[(closest(list_of_colors, color3))]))
