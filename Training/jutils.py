from enum import Enum
import os
class Colors(Enum):
    """
    ANSI escape sequences for colored text in the terminal.
    """
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"
    ORANGE = "\033[38;5;208m"
    LIGHT_BLUE = "\033[38;5;39m"
    LIGHT_GREEN = "\033[38;5;82m"
    LIGHT_YELLOW = "\033[38;5;226m"
    LIGHT_RED = "\033[38;5;196m"

def print_color(text: str, color: Colors) -> None:
    """
    Print text in the specified color.

    Args:
        text (str): The text to print.
        color (Colors): The color to print the text in.
    """
    print(f"{color.value}{text}{Colors.RESET.value}")

class ColorPrinter:
    """
    A class to handle colored printing in the terminal.
    """
    def __init__(self, color: Colors = Colors.BLUE):
        self.color = color

    def print(self, text: str, color=None) -> None:
        """
        Print text in the specified color.

        Args:
            text (str): The text to print.
            color (Colors, optional): The color to print the text in. If None, uses the instance's color.
        """
        if color is None:
            color = self.color.value
        else:
            color = color.value
        print(f"{color}{text}{Colors.RESET.value}")

def assert_file(path):
    assert os.path.exists(path), f"Path {path} does not exist"
    assert os.path.isfile(path), f"Path {path} is not a file"

def assert_dir(path):
    assert os.path.exists(path), f"Path {path} does not exist"
    assert os.path.isdir(path), f"Path {path} is not a directory"