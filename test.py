import random
# import string

file = open("morse/new_samples.txt",'w')
# random characters
# char_space = ('abcdefghijklmnopqrstuvwxyz.,?/=1234567890 ')
# TODO - figure out how to generate /? in filename for windows
# try using unicode symbols (fw ?), div / cannot be encoded so exclude it
char_space = ('abcdefghijklmnopqrstuvwxyz.,=1234567890 ')
# TODO - weighted characters
# weighted_char = 
for _ in range(27000):
    # word = ( ''.join(random.choice(string.ascii_lowercase) + random.choice(string.digits) + random.choice((' ','','','','')) for i in range (3)) ) + '\n'
    word = ( ''.join(random.choice(char_space) for i in range (random.randint(3,7))) ) + '\n'
    print(word)
    file.write(word)
file.close