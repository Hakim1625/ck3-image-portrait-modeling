import re
import json
import torch

f = open('./utils/dependencies/gene_dicts.json')
genes, non_simple_genes = json.load(f)
f.close()

def dna_to_array(path):
    with open(path, "r") as f:
        lines = f.readlines()
        new_genes = {} 

        p_gene, p_sub_gene, p_value = r'(?:\t)\w+', r'"\w+"', r'\b\d+'
        get_gene = lambda line, expression=p_gene: re.search(expression, line).group()[1:]
        get_sub_gene = lambda line, expression=p_sub_gene: re.search(expression, line).group()[1:-1]
        get_value = lambda line, expression=p_value: re.search(expression, line).group()
        


    


def array_to_dna(array, path):
    predicted_genes = dict(zip(list(genes.keys()), array.numpy().astype(int)))

    p_gene, p_sub_gene, p_value = r'(?:\t)\w+', r'"\w+"', r'\b\d+'
    get_gene = lambda line, expression=p_gene: re.search(expression, line).group()[1:]
    get_value = lambda line, expression=p_value: re.search(expression, line).group()

    get_age = lambda x: (float(x)+255)/510
    get_skincolor = lambda x: (x+255)/2

    with open('./utils/dependencies/default_dna.txt', 'r') as f1, open(path, 'w') as f2:
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

