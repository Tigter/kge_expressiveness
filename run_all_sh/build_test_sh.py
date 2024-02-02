import re

file = "./run_all_sh/rotpro_0629.sh"
result_lines = []
with open(file,'r') as f:
    lines = f.readlines()
    for line in lines:
        if len(line) < 2 :
            continue
        elif line[0] == '#':
            result_lines.append(line.strip())
        else:
            splits = line.strip().split(" ")
            path = " -init /home/skl/yl/models/RotPro_YAGO3-10_"+splits[5].strip()+"/hit10/ &"
            new_line = line.replace(" &",path)
            result_lines.append(new_line)

with open("./run_all_sh/rotpro_test.sh",'w') as f:
    for line in result_lines:
        print(line)
        f.write("%s\n" % line)