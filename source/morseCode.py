class MorseCode:
    def __init__(self, text):
        self.code = {
            "!": "-.-.--",
            "$": "...-..-",
            "'": ".----.",
            "(": "-.--.",
            ")": "-.--.-",
            ",": "--..--",
            "-": "-....-",
            ".": ".-.-.-",
            "/": "-..-.",
            "0": "-----",
            "1": ".----",
            "2": "..---",
            "3": "...--",
            "4": "....-",
            "5": ".....",
            "6": "-....",
            "7": "--...",
            "8": "---..",
            "9": "----.",
            ":": "---...",
            ";": "-.-.-.",
            ">": ".-.-.",  # <AR>
            "<": ".-...",  # <AS>
            "{": "....--",  # <HM>
            "&": "..-.-",  # <INT>
            "%": "...-.-",  # <SK>
            "}": "...-.",  # <VE>
            "=": "-...-",  # <BT>
            "?": "..--..",
            "@": ".--.-.",
            "A": ".-",
            "B": "-...",
            "C": "-.-.",
            "D": "-..",
            "E": ".",
            "F": "..-.",
            "G": "--.",
            "H": "....",
            "I": "..",
            "J": ".---",
            "K": "-.-",
            "L": ".-..",
            "M": "--",
            "N": "-.",
            "O": "---",
            "P": ".--.",
            "Q": "--.-",
            "R": ".-.",
            "S": "...",
            "T": "-",
            "U": "..-",
            "V": "...-",
            "W": ".--",
            "X": "-..-",
            "Y": "-.--",
            "Z": "--..",
            "\\": ".-..-.",
            "_": "..--.-",
            "~": ".-.-",
            " ": "_",
        }
        self.len = self.len_str(text)

    def len_dits(self, cws):
        """Return the length of CW string in dit units, including spaces."""
        val = 0
        for ch in cws:
            if ch == ".":  # dit len
                val += 1
            if ch == "-":  # dah len
                val += 3
            if ch == "_":  # word space
                val += 4
            val += 1  # el space
        val += 2  # char space = 3  (el space +2)
        return val

    def len_chr(self, ch):
        s = self.code[ch]
        # print(s)
        return self.len_dits(s)

    def len_str(self, s):
        i = 0
        for ch in s:
            val = self.len_chr(ch)
            i += val
            # print(ch, val, i)
        return i - 3  # remove last char space at end of string
