import re
import json
import torch

f = open('./utils/gene_dicts.json')
genes, non_simple_genes = json.load(f)
f.close()

def get_index(value, length):
    print(value, length)
    l = length
    for i in range(l):
        if int(value) <= 255-(2*i*255/l) and int(value) >= 255-(2*(i+1)*255/l):
            #print(value, l-i-1)
            return l-i-1

def dna_to_array(path):
    with open(path, "r") as f:
        lines = f.readlines()

        p_gene, p_sub_gene, p_value = r'(?:\t)\w+', r'"\w+"', r'\b\d+'
        get_gene = lambda line, expression=p_gene: re.search(expression, line).group()[1:]
        get_sub_gene = lambda line, expression=p_sub_gene: re.search(expression, line).group()[1:-1]
        get_value = lambda line, expression=p_value: re.search(expression, line).group()
        new_genes = {}

        get_age = lambda y: (510*float(y))-255

        for line in lines:
            if re.search(p_gene, line) != None:
                gene = get_gene(line)
                if gene in non_simple_genes.keys():
                    sub_gene = get_sub_gene(line)
                    value = get_value(line)
                    i = non_simple_genes.get(gene).index(sub_gene)
                    n = len(non_simple_genes.get(gene))
                    new_value =  -255 + ((510/n)*(i+((float(value)+255)/510)))
                    new_genes.update({gene: new_value})
                else:
                    if gene == 'age':
                        new_genes.update({'age': get_age(get_value(line, expression=r'\d+.\d+'))})
                    else:
                        if gene == 'type':
                            if "male" in line:
                                new_genes.update({'gender': -255})
                            else:
                                new_genes.update({'gender': 255})
                        else:
                            if gene == 'skin_color':
                                new_genes.update({'skincolor_light':re.findall(p_value, lines[4])[0], 'skincolor_dark':re.findall(p_value, lines[4])[1]})
                            else:
                                if gene in genes.keys():
                                    value = get_value(line)
                                    if 'neg' in new_genes:  
                                        new_genes.update({gene: int(value)*-1})
                                    else:
                                        new_genes.update({gene: int(value)})
    return torch.tensor([float(v) for v in new_genes.values()])
    


def array_to_dna(array, path):
    predicted_genes = dict(zip(list(genes.keys()), array.numpy().astype(int)))
    #[print(item) for item in predicted_genes.items()]
    #[print(item) for item in non_simple_genes.items()]
    

    p_gene, p_sub_gene, p_value = r'(?:\t)\w+', r'"\w+"', r'\b\d+'
    get_gene = lambda line, expression=p_gene: re.search(expression, line).group()[1:]
    get_value = lambda line, expression=p_value: re.search(expression, line).group()

    get_age = lambda x: (float(x)+255)/510

    with open('./utils/default_dna.txt', 'r') as f1, open(path, 'w') as f2:
        new_lines=[]
        lines = f1.readlines()
        
        for line in lines:
            subline = line
            if re.search(p_gene, line) != None:
                gene = get_gene(line)
                

                if gene == 'age':
                    new_lines.append(line.replace(get_value(line, expression=r'\d+.\d+'), str(get_age(predicted_genes[gene]))))
                else:
                    if gene == 'type':
                        if float(predicted_genes['gender']) > 0:
                            new_lines.append(line.replace(re.search(r'=\w+', line).group()[1:], 'female'))
                        else:
                            new_lines.append(line.replace(re.search(r'=\w+', line).group()[1:], 'male'))
                    else:
                        if 'skin_color' in line:
                            substring = re.findall(r'\b(\d+\s\d+)\b', subline)[0]
                   
                            subline = subline.replace(substring, '{} {} '.format(predicted_genes['skincolor_light'], predicted_genes['skincolor_dark']))

                            new_lines.append(subline)
                        else:
                            if gene in non_simple_genes.keys():
                                    value =  predicted_genes[gene]
                                    length = len(non_simple_genes[gene])
                                    index = get_index(value, length)
                                    sub_gene = non_simple_genes[gene][index]
                                    new_value = length*(value+255)-(510*(index))-255
                                    #print(gene, sub_gene, length, index, value, new_value)
                                    for substring in re.findall(r'"\w+"\s\d+', line):
                                        subline = subline.replace(substring, '"{}" {}'.format(sub_gene, new_value))
                                    new_lines.append(subline)
                            else:
                                if gene in genes.keys():
                                    if float(predicted_genes[gene]) > 0:
                                        for substring in re.findall(r'"\w+"\s\d+', line):
                                            subline = subline.replace(substring, '{}pos" {}'.format(re.search(r'"\w+_', line)[0], predicted_genes[gene]))
                                        new_lines.append(subline)
                                    else:
                                        for substring in re.findall(r'"\w+"\s\d+', line):
                                            subline= subline.replace(substring, '{}neg" {}'.format(re.search(r'"\w+_', line)[0], predicted_genes[gene]*-1))
                                        new_lines.append(subline)
                                else:
                                    new_lines.append(line)
            else:
                new_lines.append(line)

        f1.close()
        f2.writelines(new_lines)
        f2.close()

def main():
    array = torch.randn(101).uniform_(-255, 255)
    array_to_dna(array, 'dna.txt')
    

    #get_index(-255, 3)
    
    #get_index(-85, 3)
    
    #get_index(84, 3)
    
    #get_index(255, 3)

if __name__ == "__main__":
    main()

import re
import json
import torch

f = open('./utils/gene_dicts.json')
genes, non_simple_genes = json.load(f)
f.close()

def get_index(value, length):
    print(value, length)
    l = length
    for i in range(l):
        if int(value) <= 255-(2*i*255/l) and int(value) >= 255-(2*(i+1)*255/l):
            #print(value, l-i-1)
            return l-i-1

def dna_to_array(path):
    with open(path, "r") as f:
        lines = f.readlines()

        p_gene, p_sub_gene, p_value = r'(?:\t)\w+', r'"\w+"', r'\b\d+'
        get_gene = lambda line, expression=p_gene: re.search(expression, line).group()[1:]
        get_sub_gene = lambda line, expression=p_sub_gene: re.search(expression, line).group()[1:-1]
        get_value = lambda line, expression=p_value: re.search(expression, line).group()
        new_genes = {}

        get_age = lambda y: (510*float(y))-255

        for line in lines:
            if re.search(p_gene, line) != None:
                gene = get_gene(line)
                if gene in non_simple_genes.keys():
                    sub_gene = get_sub_gene(line)
                    value = get_value(line)
                    i = non_simple_genes.get(gene).index(sub_gene)
                    n = len(non_simple_genes.get(gene))
                    new_value =  -255 + ((510/n)*(i+((float(value)+255)/510)))
                    new_genes.update({gene: new_value})
                else:
                    if gene == 'age':
                        new_genes.update({'age': get_age(get_value(line, expression=r'\d+.\d+'))})
                    else:
                        if gene == 'type':
                            if "male" in line:
                                new_genes.update({'gender': -255})
                            else:
                                new_genes.update({'gender': 255})
                        else:
                            if gene == 'skin_color':
                                new_genes.update({'skincolor_light':re.findall(p_value, lines[4])[0], 'skincolor_dark':re.findall(p_value, lines[4])[1]})
                            else:
                                if gene in genes.keys():
                                    value = get_value(line)
                                    if 'neg' in new_genes:  
                                        new_genes.update({gene: int(value)*-1})
                                    else:
                                        new_genes.update({gene: int(value)})
    return torch.tensor([float(v) for v in new_genes.values()])
    


def array_to_dna(array, path):
    predicted_genes = dict(zip(list(genes.keys()), array.numpy().astype(int)))
    #[print(item) for item in predicted_genes.items()]
    #[print(item) for item in non_simple_genes.items()]
    

    p_gene, p_sub_gene, p_value = r'(?:\t)\w+', r'"\w+"', r'\b\d+'
    get_gene = lambda line, expression=p_gene: re.search(expression, line).group()[1:]
    get_value = lambda line, expression=p_value: re.search(expression, line).group()

    get_age = lambda x: (float(x)+255)/510

    with open('./utils/default_dna.txt', 'r') as f1, open(path, 'w') as f2:
        new_lines=[]
        lines = f1.readlines()
        
        for line in lines:
            subline = line
            if re.search(p_gene, line) != None:
                gene = get_gene(line)
                

                if gene == 'age':
                    new_lines.append(line.replace(get_value(line, expression=r'\d+.\d+'), str(get_age(predicted_genes[gene]))))
                else:
                    if gene == 'type':
                        if float(predicted_genes['gender']) > 0:
                            new_lines.append(line.replace(re.search(r'=\w+', line).group()[1:], 'female'))
                        else:
                            new_lines.append(line.replace(re.search(r'=\w+', line).group()[1:], 'male'))
                    else:
                        if 'skin_color' in line:
                            substring = re.findall(r'\b(\d+\s\d+)\b', subline)[0]
                   
                            subline = subline.replace(substring, '{} {} '.format(predicted_genes['skincolor_light'], predicted_genes['skincolor_dark']))

                            new_lines.append(subline)
                        else:
                            if gene in non_simple_genes.keys():
                                    value =  predicted_genes[gene]
                                    length = len(non_simple_genes[gene])
                                    index = get_index(value, length)
                                    sub_gene = non_simple_genes[gene][index]
                                    new_value = length*(value+255)-(510*(index))-255
                                    #print(gene, sub_gene, length, index, value, new_value)
                                    for substring in re.findall(r'"\w+"\s\d+', line):
                                        subline = subline.replace(substring, '"{}" {}'.format(sub_gene, new_value))
                                    new_lines.append(subline)
                            else:
                                if gene in genes.keys():
                                    if float(predicted_genes[gene]) > 0:
                                        for substring in re.findall(r'"\w+"\s\d+', line):
                                            subline = subline.replace(substring, '{}pos" {}'.format(re.search(r'"\w+_', line)[0], predicted_genes[gene]))
                                        new_lines.append(subline)
                                    else:
                                        for substring in re.findall(r'"\w+"\s\d+', line):
                                            subline= subline.replace(substring, '{}neg" {}'.format(re.search(r'"\w+_', line)[0], predicted_genes[gene]*-1))
                                        new_lines.append(subline)
                                else:
                                    new_lines.append(line)
            else:
                new_lines.append(line)

        f1.close()
        f2.writelines(new_lines)
        f2.close()

def main():
    array = torch.randn(101).uniform_(-255, 255)
    array_to_dna(array, 'dna.txt')
    

    #get_index(-255, 3)
    
    #get_index(-85, 3)
    
    #get_index(84, 3)
    
    #get_index(255, 3)

if __name__ == "__main__":
    main()
