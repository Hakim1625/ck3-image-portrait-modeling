import re
import json
import torch
import torch.nn.functional as F

def load_dict(path):
    f = open(path)
    genes = json.load(f)
    f.close()
    return genes

def verify_gender(gender):
    if gender == 'male': return

def dna_ToTensor(path, genes=load_dict('./utils/dependencies/gene_dicts.json')):
    tensors = []
    with open(path, "r") as f:
        lines = f.readlines()

        p_gene, p_sub_gene, p_value, p_age, p_gender = r'(?:\t)\w+', r'"\w+"', r'\b\d+', r'\d.\d+', r'[=]\w+'
        get_gene = lambda line, expression=p_gene: re.search(expression, line).group()[1:]
        get_sub_gene = lambda line, expression=p_sub_gene: re.search(expression, line).group()[1:-1]
        get_value = lambda line, expression=p_value: re.search(expression, line).group()

        for line in lines:
            try: gene, sub_gene = get_gene(line), get_sub_gene(line)
            except: 
                if 'age' in line:
                    gene = 'age'
                    sub_gene = 'age'
                    value = float(get_value(line, expression=p_age))*255
                  
                    index = genes[gene].index(sub_gene)
                    length = len(genes[gene])

                    tensor = F.one_hot(torch.tensor([index]), num_classes=length)
                    tensors.append(value*tensor)

                if 'type' in line: 
                    gene = 'gender'
                    value = 255
                    sub_gene = sub_gene = get_gene(line, expression=p_gender)

                    index = genes[gene].index(sub_gene)
                    length = len(genes[gene])

                    tensor = F.one_hot(torch.tensor([index]), num_classes=length)
                    tensors.append(value*tensor)
                
                continue

            if gene in list(genes.keys()):
                    value = float(get_value(line))

                    index = genes[gene].index(sub_gene)
                    length = len(genes[gene])

                    tensor = F.one_hot(torch.tensor([index]), num_classes=length)
                    tensors.append(value*tensor)


    tensors = tuple(tensors)
    tensors = torch.cat(tensors, dim=1)
    return torch.squeeze(tensors)

def tensor_ToDna(tensor, path, genes=load_dict('./utils/dependencies/gene_dicts.json')):
    target_genes = {}

    lengths = [len(sub_genes) for sub_genes in list(genes.values())]

    for i, gene in enumerate(genes):

        i_2 = lengths[i]
        i_1 = sum(lengths[:i])

        sub_tensor = tensor[i_1:i_1+i_2]
        index = torch.argmax(sub_tensor).item()

        value = torch.max(sub_tensor).item()
        sub_gene = genes[gene][index]

        target_genes.update({gene: {'sub_gene': sub_gene, 'value': value}})


    with open('./utils/dependencies/default_dna.txt', 'r') as f1, open(path, 'w') as f2:
        new_lines=[]
        lines = f1.readlines()

        p_gene = r'(?:\t)\w+'
        get_gene = lambda line, expression=p_gene: re.search(expression, line).group()[1:]

        for line in lines:
            try: gene = get_gene(line)
            except: continue
            
            if gene in list(target_genes.keys()):
                sub_gene = target_genes[gene]['sub_gene']
                value = target_genes[gene]['value']
                sub_line = line
                for substring in re.findall(r'"\w+"\s\d+', line):
                    sub_line = sub_line.replace(substring, '"{}" {}'.format(sub_gene, value))

                new_lines.append(sub_line)

            else:
                new_lines.append(line)

        f1.close()
        f2.writelines(new_lines)
        f2.close()

if __name__ == '__main__':
    #tensor_ToDna(torch.randn(220), './test.txt')
    print(dna_ToTensor('portraits_embeddings/6/6.txt').size())