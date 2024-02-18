total_lines = []
languages = ['en', 'cs', 'sk', 'tr', 'hu']
for language in languages:
    with open(f'data/{language}.txt', 'r') as f:
        lines = f.readlines()
        total_lines += lines

#for line in lines:
#    print(line.strip())

with open('data/all_lines.txt', 'w') as f:
    for line in total_lines:
        f.write(line)

