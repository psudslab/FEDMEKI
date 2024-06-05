#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:08:16 2024

@author: xmw5190
"""

import sys
print(sys.executable)

import timm
import inspect

# Assuming MyClass is the class you're interested in


# Get the module name
module_name = timm.models.vision_transformer.VisionTransformer.__module__

# Import the module dynamically using __import__ function
module = __import__(module_name)

# Use inspect.getfile to get the file path
file_path = inspect.getfile(module)

print(f"The class is defined in: {file_path}")