"""Project config

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

params = {
        'epoch_offset': 0,
        'classes' : ["with_mask", "without_mask", "mask_weared_incorrect"],
        }

# aspect ratios
def anchor_aspect_ratios():
    aspect_ratios = config.params['aspect_ratios']
    return aspect_ratios

