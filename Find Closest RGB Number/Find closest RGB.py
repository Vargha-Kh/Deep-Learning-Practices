from math import sqrt
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


def closest_color(rgb):
    r, g, b = rgb
    color_difference_list = []
    for color in colors_dict:
        cr, cg, cb = color
        color_diff = sqrt(abs(r - cr) ** 2 + abs(g - cg) ** 2 + abs(b - cb) ** 2)
        color_difference_list.append((color_diff, color))
    return min(color_difference_list)


print("The Distance        Closest Color")
print(colored(closest_color((12, 34, 156)), colors_dict[(closest_color((12, 34, 156))[1])]))
print(colored(closest_color((189, 240, 100)), colors_dict[(closest_color((189, 240, 100))[1])]))
print(colored(closest_color((240, 25, 40)), colors_dict[(closest_color((240, 25, 40))[1])]))
print(colored(closest_color((200, 230, 270)), colors_dict[(closest_color((200, 230, 270))[1])]))
