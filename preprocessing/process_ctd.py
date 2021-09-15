import argparse
import csv
import json
import traceback
import sys

csv.field_size_limit(sys.maxsize)

import os

def save_pid2data(pid2data, output_dir):
    for pid in pid2data:
        data = pid2data[pid]
        save_path = os.path.join(output_dir, f"{pid}.jsonl")
        with open(save_path, 'w') as fo:
            for sample in data:
                new_sample ={
                    'predicate_id':pid,
                    'sub_uri': sample['sub_uri'],
                    'sub_label': sample['sub_label'],
                    'sub_aliases': sample['sub_aliases'],
                    'obj_uri': sample['obj_uri'],
                    'obj_label': sample['obj_label'],
                    'obj_aliases': sample['obj_aliases'],
                }
                new_sample = json.dumps(new_sample, ensure_ascii=False)
                fo.write(new_sample + "\n")

def process_chemicals_genes_ixns(data_file, output_dir, chemicals_dict, genes_dict):
    print(f"Opening {data_file}")
    data = []
    properties2meta = {}
    with open(file=data_file) as f:
        tsv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        CG = 1
        for line in tsv_reader:
            if line[0].startswith("#"):
                continue
            ChemicalName, ChemicalID, CasRN, GeneSymbol, GeneID, GeneForms, Organism, OrganismID, Interaction, InteractionActions, PubMedIDs = line

            if Organism != 'Homo sapiens':
                continue
            # use unique interaction
            if '|' in InteractionActions:
                continue
            # use unique gene type
            if ('|' in GeneForms) or (GeneForms == ''):
                continue
            
            if chemicals_dict[ChemicalID]['name'] != ChemicalName:
                import pdb ; pdb.set_trace()
            if  GeneID not in genes_dict: # not human gene
                # print(f"{GeneID} not in dict => Skip")
                continue
            elif genes_dict[GeneID]['symbol'].upper() != GeneSymbol.upper():
                import pdb ; pdb.set_trace()
            else:
                gene_synonyms = [genes_dict[GeneID]['name']] + genes_dict[GeneID]['synonyms']
            
            InteractionActions = InteractionActions.replace("^"," ")

            property_label = InteractionActions + "_" + GeneForms
            data.append({
                'property_label':property_label,
                'evidence': Interaction,
                'sub_uri': ChemicalID,
                'sub_label': ChemicalName,
                'sub_aliases': chemicals_dict[ChemicalID]['synonyms'],
                'obj_uri': GeneID,
                'obj_label': GeneSymbol,
                'obj_aliases': gene_synonyms
            })

            if property_label not in properties2meta:
                properties2meta[property_label] = {
                    'count':0,
                    'pid':f'CG{CG}'
                }
                CG += 1

                properties2meta[property_label]['property'] = InteractionActions
                properties2meta[property_label]['obj_type'] = GeneForms
                properties2meta[property_label]['example'] = Interaction
            
            properties2meta[property_label]['count'] += 1

    pid2data = {}
    for sample in data:
        _prop = sample['property_label']
        meta = properties2meta[_prop]
        pid = meta['pid']
        count = meta['count']

        # filter less than 2000
        if count < 2000:
            continue
        if pid not in pid2data:
            pid2data[pid] = []
        
        pid2data[pid].append(sample)

    save_pid2data(pid2data, output_dir)
    
def process_chemicals_diseases(data_file, output_dir, chemicals_dict, diseases_dict):
    print(f"Opening {data_file}")
    data = []
    properties2meta = {}
    with open(file=data_file) as f:
        tsv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        CD = 1
        for line in tsv_reader:
            if line[0].startswith("#"):
                continue
            ChemicalName, ChemicalID, CasRN, DiseaseName, DiseaseID, DirectEvidence, InferenceGeneSymbol, InferenceScore,OmimIDs, PubMedIDs = line

            DiseaseID = DiseaseID.replace("MESH:", "")
            # filter infered
            if not DirectEvidence:
                continue
            
            property_label = DirectEvidence
            # print(property_label)
            data.append({
                'property_label':property_label,
                'sub_uri': ChemicalID,
                'sub_label': ChemicalName,
                'sub_type': 'chemical',
                'sub_aliases': chemicals_dict[ChemicalID]['synonyms'],
                'obj_uri': DiseaseID,
                'obj_label': DiseaseName,
                'obj_type': 'disease',
                'obj_aliases': diseases_dict[DiseaseID]['synonyms']
            })

            if property_label not in properties2meta:
                properties2meta[property_label] = {
                    'count':0,
                    'pid':f'CD{CD}'
                }
                CD += 1

                properties2meta[property_label]['property'] = property_label
            
            properties2meta[property_label]['count'] += 1

    pid2data = {}
    for sample in data:
        _prop = sample['property_label']
        meta = properties2meta[_prop]
        pid = meta['pid']
        count = meta['count']

        # filter less than 2000
        if count < 2000:
            continue
        if pid not in pid2data:
            pid2data[pid] = []
        
        pid2data[pid].append(sample)

    save_pid2data(pid2data, output_dir)

def process_genes_diseases(data_file, output_dir, genes_dict, diseases_dict):
    print(f"Opening {data_file}")
    data = []
    properties2meta = {}
    with open(file=data_file) as f:
        tsv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        GD = 1
        for line in tsv_reader:
            if line[0].startswith("#"):
                continue
            GeneSymbol, GeneID, DiseaseName, DiseaseID, DirectEvidence, InferenceChemicalName, InferenceScore, OmimIDs, PubMedIDs = line

            DiseaseID = DiseaseID.replace("MESH:", "")
            # filter infered
            if not DirectEvidence:
                continue
            if  GeneID not in genes_dict: # not human gene
                # print(f"{GeneID} not in dict => Skip")
                continue
            property_label = DirectEvidence
            # print(property_label)
            data.append({
                'property_label':property_label,
                'sub_uri': GeneID,
                'sub_label': GeneSymbol,
                'sub_type': 'gene',
                'sub_aliases': [genes_dict[GeneID]['name']] + genes_dict[GeneID]['synonyms'],
                'obj_uri': DiseaseID,
                'obj_label': DiseaseName,
                'obj_type': 'disease',
                'obj_aliases': diseases_dict[DiseaseID]['synonyms']
            })

            if property_label not in properties2meta:
                properties2meta[property_label] = {
                    'count':0,
                    'pid':f'GD{GD}'
                }
                GD += 1

                properties2meta[property_label]['property'] = property_label
            
            properties2meta[property_label]['count'] += 1

    pid2data = {}
    for sample in data:
        _prop = sample['property_label']
        meta = properties2meta[_prop]
        pid = meta['pid']
        count = meta['count']

        # filter less than 2000
        if count < 2000:
            continue
        if pid not in pid2data:
            pid2data[pid] = []
        
        pid2data[pid].append(sample)

    save_pid2data(pid2data, output_dir)

def process_genes_pathways(data_file, output_dir, genes_dict):
    print(f"Opening {data_file}")
    data = []
    properties2meta = {}
    with open(file=data_file) as f:
        tsv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        GP = 1
        for line in tsv_reader:
            if line[0].startswith("#"):
                continue
            GeneSymbol, GeneID, PathwayName, PathwayID = line

            property_label = 'association'
            if  GeneID not in genes_dict: # not human gene
                # print(f"{GeneID} not in dict => Skip")
                continue
            
            data.append({
                'property_label':property_label,
                'sub_uri': GeneID,
                'sub_label': GeneSymbol,
                'sub_type': 'gene',
                'sub_aliases': [genes_dict[GeneID]['name']] + genes_dict[GeneID]['synonyms'],
                'obj_uri': PathwayID,
                'obj_label': PathwayName,
                'obj_type':'pathway',
                'obj_aliases': []
            })

            if property_label not in properties2meta:
                properties2meta[property_label] = {
                    'count':0,
                    'pid':f'GP{GP}'
                }
                GP += 1

                properties2meta[property_label]['property'] = property_label
            
            properties2meta[property_label]['count'] += 1

    pid2data = {}
    for sample in data:
        _prop = sample['property_label']
        meta = properties2meta[_prop]
        pid = meta['pid']
        count = meta['count']

        # filter less than 2000
        if count < 2000:
            continue
        if pid not in pid2data:
            pid2data[pid] = []
        
        pid2data[pid].append(sample)

    save_pid2data(pid2data, output_dir)

def process_chemicals_phenotypestype(data_file, output_dir, chemicals_dict):
    print(f"Opening {data_file}")
    data = []
    properties2meta = {}
    with open(file=data_file) as f:
        tsv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        CP = 1
        for line in tsv_reader:
            if line[0].startswith("#"):
                continue
            chemicalname, chemicalid,casrn,phenotypename,phenotypeid,comentionedterms,organism,organismid,interaction,interactionactions,anatomyterms,inferencegenesymbols,pubmedids, _ = line
            if organism != 'Homo sapiens':
                continue
            # use unique interaction
            if '|' in interactionactions:
                continue
            
            assert chemicals_dict[chemicalid]['name'] == chemicalname

            interactionactions = interactionactions.replace("^"," ")

            property_label = interactionactions

            data.append({
                'property_label':property_label,
                'sub_uri': chemicalid,
                'sub_label': chemicalname,
                'sub_type': 'chemical',
                'sub_aliases': chemicals_dict[chemicalid]['synonyms'],
                'obj_uri': phenotypeid,
                'obj_label': phenotypename,
                'obj_type': 'phenotype',
                'obj_aliases': []
            })

            if property_label not in properties2meta:
                properties2meta[property_label] = {
                    'count':0,
                    'pid':f'CP{CP}'
                }
                CP += 1

                properties2meta[property_label]['property'] = property_label
                properties2meta[property_label]['example'] = interaction
            
            properties2meta[property_label]['count'] += 1

    pid2data = {}
    for sample in data:
        _prop = sample['property_label']
        meta = properties2meta[_prop]
        pid = meta['pid']
        count = meta['count']

        # filter less than 2000
        if count < 2000:
            continue
        if pid not in pid2data:
            pid2data[pid] = []
        
        pid2data[pid].append(sample)

    save_pid2data(pid2data, output_dir)

def process_genes_dict(data_file):
    print(f"Opening {data_file}")
    geneid2meta = {}
    with open(file=data_file) as f:
        tsv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        for line in tsv_reader:
            if line[0].startswith("#"):
                continue

            GeneSymbol, GeneName, GeneID, AltGeneIDs, Synonyms, BioGRIDIDs, PharmGKBIDs, UniProtIDs = line
            geneid2meta[GeneID] = {
                'symbol': GeneSymbol,
                'name': GeneName,
                'synonyms': Synonyms.split("|")
            }
    
    print(f"len(geneid2meta)={len(geneid2meta)}")
    return geneid2meta

def process_chemicals_dict(data_file):
    print(f"Opening {data_file}")
    chemid2meta = {}
    with open(file=data_file) as f:
        tsv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        for line in tsv_reader:
            if line[0].startswith("#"):
                continue

            ChemicalName,ChemicalID,CasRN,Definition,ParentIDs,TreeNumbers,ParentTreeNumbers,Synonyms = line
            ChemicalID = ChemicalID.replace("MESH:","")
            try:
                chemid2meta[ChemicalID] = {
                    'name': ChemicalName,
                    'synonyms': Synonyms.split("|")
                }
            except Exception as e:
                print(e)
                traceback.print_exc()
                raise e

    
    print(f"len(chemid2meta)={len(chemid2meta)}")
    return chemid2meta

def process_diseases_dict(data_file):
    print(f"Opening {data_file}")
    diseaseid2meta = {}
    with open(file=data_file) as f:
        tsv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        for line in tsv_reader:
            if line[0].startswith("#"):
                continue

            DiseaseName, DiseaseID, AltDiseaseIDs, Definition, ParentIDs, TreeNumbers, ParentTreeNumbers, Synonyms, SlimMappings = line
            DiseaseID = DiseaseID.replace("MESH:","")
            try:
                diseaseid2meta[DiseaseID] = {
                    'name': DiseaseName,
                    'synonyms': Synonyms.split("|")
                }
            except Exception as e:
                print(e)
                traceback.print_exc()
                import pdb ; pdb.set_trace()
    
    print(f"len(diseaseid2meta)={len(diseaseid2meta)}")
    return diseaseid2meta

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.chemicals_dict:
        chemicals_dict = process_chemicals_dict(data_file=args.chemicals_dict)
    else:
        print("[WARN] no args.chemicals_dict")

    if args.diseases_dict:
        diseases_dict = process_diseases_dict(data_file=args.diseases_dict)
    else:
        print("[WARN] no args.diseases_dict")

    # use preprocessed human gene dictionary
    if args.genes_dict and args.genes_dict.endswith(".json"):
        with open(args.genes_dict) as f:
            genes_dict = json.load(f)
    else:
        print("[WARN] no args.genes_dict")
    
    if args.chemicals_genes_file:
        process_chemicals_genes_ixns(
            data_file=args.chemicals_genes_file,
            output_dir=args.output_dir,
            chemicals_dict=chemicals_dict,
            genes_dict=genes_dict
        )
    else:
        print("[WARN] no args.chemicals_genes_file")
    
    if args.chemicals_diseases_file:
        process_chemicals_diseases(
            data_file=args.chemicals_diseases_file,
            output_dir=args.output_dir,
            chemicals_dict=chemicals_dict,
            diseases_dict=diseases_dict
        )
    else:
        print("[WARN] no args.chemicals_diseases_file")

    if args.genes_diseases_file:
        process_genes_diseases(
            data_file=args.genes_diseases_file,
            output_dir=args.output_dir,
            genes_dict=genes_dict,
            diseases_dict=diseases_dict
        )

    if args.genes_pathways_file:
        process_genes_pathways(
            data_file=args.genes_pathways_file,
            output_dir=args.output_dir,
            genes_dict=genes_dict
        )

    if args.chemicals_phenotypes_file:
        process_chemicals_phenotypestype(
            data_file=args.chemicals_phenotypes_file,
            output_dir=args.output_dir,
            chemicals_dict=chemicals_dict
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chemicals_dict', type=str)
    parser.add_argument('--diseases_dict',  type=str)
    parser.add_argument('--genes_dict', type=str)
    parser.add_argument('--chemicals_genes_file',  type=str)
    parser.add_argument('--chemicals_diseases_file',  type=str)
    parser.add_argument('--chemicals_phenotypes_file', type=str)
    parser.add_argument('--genes_diseases_file', type=str)
    parser.add_argument('--genes_pathways_file', type=str)
    parser.add_argument('--output_dir',  type=str)

    args = parser.parse_args()

    main(args)
