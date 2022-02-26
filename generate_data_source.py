import random

# import string

file = open("morse/weighted_samples.txt", "w")
# random characters
# char_space = ('abcdefghijklmnopqrstuvwxyz.,?/=1234567890 ')
# TODO - figure out how to generate /? in filename for windows
# char_space = ('abcdefghijklmnopqrstuvwxyz.,=1234567890 ')
char_space = (
    "a" * 63
    + "b" * 11
    + "c" * 22
    + "d" * 33
    + "e" * 98
    + "f" * 17
    + "g" * 16
    + "h" * 47
    + "i" * 54
    + "j" * 1
    + "k" * 6
    + "l" * 31
    + "m" * 19
    + "n" * 52
    + "o" * 58
    + "p" * 15
    + "q" * 1
    + "r" * 46
    + "s" * 49
    + "t" * 70
    + "u" * 21
    + "v" * 8
    + "w" * 18
    + "x" * 1
    + "y" * 15
    + "z" * 1
    + "." * 9
    + "," * 7
    + "=" * 10
    + "1" * 3
    + "2" * 3
    + "3" * 3
    + "4" * 3
    + "5" * 3
    + "6" * 3
    + "7" * 3
    + "8" * 3
    + "9" * 3
    + "0" * 3
    + " " * 172
)
print(char_space)
# TODO - more accurately weighted characters
for _ in range(27000):
    # word = ( ''.join(random.choice(string.ascii_lowercase) 
    # + random.choice(string.digits) 
    # + random.choice((' ','','','','')) 
    # for i in range (3)) ) + '\n'
    word = (
        "".join(random.choice(char_space) for i in range(random.randint(4, 7)))
    ) + "\n"
    print(word)
    file.write(word)
file.close
