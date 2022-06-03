import random

file = open("source/samples_chars.txt", "w")
# random characters
# char_space = ('abcdefghijklmnopqrstuvwxyz.,?/=1234567890 ')
# TODO - figure out how to generate /? in filename for windows
char_space = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.', ',', '=', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',  # 'AR'
              ]
char_weights = (.06314691, .01153608, 0.021510309, 0.032884021, 0.09821134, 0.017226804, 0.015579897, 0.047118557,  # abcdefgh
                0.053860825, 0.00118299, 0.005969072, 0.031121134, 0.018603093, 0.05218299, 0.058043814, 0.014914948,   # ijklmnop
                0.000734536, 0.046291237, 0.048920103, 0.070020619, 0.021324742, 0.007561856, 0.018247423, 0.001159794,  # qrstuvwx
                0.015262887, 0.000572165, 0.171821306, 0.008591065, 0.006872852, 0.008083333,  # 0.003436426, 0.000515464, # ? and /, z .,=
                0.002749141, 0.002749141, 0.002749141, 0.002749141, 0.002749141, 0.002749141, 0.002749141, 0.002749141, # 12345678
                0.002749141, 0.002749141) # 90
print(char_space)
for _ in range(27000):
    word = "".join(random.choices(char_space, char_weights, k=1)) + '\n'
    # prevent beginning or ending with space
    while (word[0] == ' '):
        replace = random.choices(char_space, char_weights)
        word = replace[0] + word[1:]
    while (word[-1] == ' '):
        replace = random.choices(char_space, char_weights)
        word = word[:-3] + replace[0]  # + "\n"
    # print(word)
    file.write(word)
file.close
