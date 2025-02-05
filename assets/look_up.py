from local_pm4py.functions import declare_processing as dp
from local_pm4py.functions.declare_processing import g
from automata.fa.nfa import NFA
from automata.fa.dfa import DFA
import pandas as pd
import json

def check_all(S1,S2,SS1, SS2, dfa_decl, cut_type, S_mapped):
    flag = False
    dev_list = []
    if cut_type == 'seq':
        for p in SS1:
            for q in SS2:
                reg_to_check1 = f"{g(S1 - {p})}*{p}{g(S1 - {p})}*{g(S2 - {q})}*{q}{g(S2 - {q})}*"
                nfa1 = NFA.from_regex(reg_to_check1, input_symbols=set(S_mapped.values()))
                dfa_to_check1 = dp.rename_dfa_transitions(DFA.from_nfa(nfa1),S_mapped)
                if dfa_to_check1.issubset(dfa_decl.complement()):
                    flag = True
                    dev_list.append(reg_to_check1)
                    # block = True


    elif cut_type == 'exc':
        for p in SS1:
            reg_to_check2 = f"{g(S1 - {p})}*{p}{g(S1 - {p})}*"
            nfa2 = NFA.from_regex(reg_to_check2, input_symbols=set(S_mapped.values()))
            dfa_to_check2 = dp.rename_dfa_transitions(DFA.from_nfa(nfa2),S_mapped)
            if dfa_to_check2.issubset(dfa_decl.complement()):
                flag = True
                dev_list.append(reg_to_check2)
                # block = True
        for q in SS2:
            reg_to_check3 = f"{g(S2 - {q})}*{q}{g(S2 - {q})}*"
            nfa3 = NFA.from_regex(reg_to_check3, input_symbols=set(S_mapped.values()))
            dfa_to_check3 = dp.rename_dfa_transitions(DFA.from_nfa(nfa3),S_mapped)
            if dfa_to_check3.issubset(dfa_decl.complement()):
                # print(f"q is {q}")
                flag = True
                dev_list.append(reg_to_check3)


    elif cut_type == 'par':
        for p in SS1:
            for q in SS2:
                reg_to_check4 = f"{g(S1.union(S2) - {p, q})}*{p}{g(S1.union(S2) - {p, q})}*{q}{g(S1.union(S2) - {p, q})}*"
                reg_to_check5 = f"{g(S1.union(S2) - {p, q})}*{q}{g(S1.union(S2) - {p, q})}*{p}{g(S1.union(S2) - {p, q})}*"
                nfa4 = NFA.from_regex(reg_to_check4, input_symbols=set(S_mapped.values()))
                dfa_to_check4 = dp.rename_dfa_transitions(DFA.from_nfa(nfa4),S_mapped)
                nfa5 = NFA.from_regex(reg_to_check5, input_symbols=set(S_mapped.values()))
                dfa_to_check5 = dp.rename_dfa_transitions(DFA.from_nfa(nfa5),S_mapped)
                if dfa_to_check4.issubset(dfa_decl.complement()):
                    flag = True
                    dev_list.append(reg_to_check4)
                    # block = True

                if dfa_to_check5.issubset(dfa_decl.complement()):
                    flag = True
                    dev_list.append(reg_to_check5)


    elif cut_type == 'loop':
        for p in SS1:
            reg_to_check6 = f"{g(S1-{p})}*{p}{g(S1-{p})}*"
            nfa6 = NFA.from_regex(reg_to_check6, input_symbols=set(S_mapped.values()))
            dfa_to_check6 = dp.rename_dfa_transitions(DFA.from_nfa(nfa6), S_mapped)
            if dfa_to_check6.issubset(dfa_decl.complement()):
                flag = True
                dev_list.append(reg_to_check6)

            for pp in SS1:
                reg_to_check9 = f"({g(S1-{p})}*{p}{g(S1-{p})}*)({g(S2)}*({g(S1-{pp})}*{pp}{g(S1-{pp})}*)){{1,1}}"
                nfa9 = NFA.from_regex(reg_to_check9, input_symbols=set(S_mapped.values()))
                dfa_to_check9 = dp.rename_dfa_transitions(DFA.from_nfa(nfa9), S_mapped)
                if dfa_to_check9.issubset(dfa_decl.complement()):
                    flag = True
                    dev_list.append(reg_to_check9)
            reg_to_check9_2 = f"({g(S1 - {p})}*{p}{g(S1 - {p})}*)({g(S2)}*({g(S1 - {p})}*{p}{g(S1 - {p})}*)){{2,2}}"
            nfa9_2 = NFA.from_regex(reg_to_check9_2, input_symbols=set(S_mapped.values()))
            dfa_to_check9_2 = dp.rename_dfa_transitions(DFA.from_nfa(nfa9_2), S_mapped)
            if dfa_to_check9_2.issubset(dfa_decl.complement()):
                flag = True
                dev_list.append(reg_to_check9_2)
            reg_to_check9_3 = f"({g(S1 - {p})}*{p}{g(S1 - {p})}*)({g(S2)}*({g(S1 - {p})}*{p}{g(S1 - {p})}*)){{3,3}}"
            nfa9_3 = NFA.from_regex(reg_to_check9_3, input_symbols=set(S_mapped.values()))
            dfa_to_check9_3 = dp.rename_dfa_transitions(DFA.from_nfa(nfa9_3), S_mapped)
            if dfa_to_check9_3.issubset(dfa_decl.complement()):
                flag = True
                dev_list.append(reg_to_check9_3)

            for q in SS2:
                reg_to_check7 = f"({g(S1-{p})}*{p}{g(S1-{p})}*)(({g(S2-{q})}*{q}{g(S2-{q})}*){g(S1)}*)+"
                nfa7 = NFA.from_regex(reg_to_check7, input_symbols=set(S_mapped.values()))
                dfa_to_check7 = dp.rename_dfa_transitions(DFA.from_nfa(nfa7),S_mapped)
                if dfa_to_check7.issubset(dfa_decl.complement()):
                    flag = True
                    dev_list.append(reg_to_check7)

                reg_to_check8 = f"{g(S1)}*(({g(S2 - {q})}*{q}{g(S2 - {q})}*)({g(S1-{p})}*{p}{g(S1-{p})}*))+"
                nfa8 = NFA.from_regex(reg_to_check8, input_symbols=set(S_mapped.values()))
                dfa_to_check8 = dp.rename_dfa_transitions(DFA.from_nfa(nfa8), S_mapped)
                if dfa_to_check8.issubset(dfa_decl.complement()):
                    flag = True
                    dev_list.append(reg_to_check8)

                for qq in SS2:
                    reg_to_check10 = f"{g(S1)}*({g(S2-{q})}*{q}{g(S2-{q})}*){g(S1)}*(({g(S2-{qq})}*{qq}{g(S2-{qq})}*){g(S1)}*){{1,1}}"
                    nfa10 = NFA.from_regex(reg_to_check10, input_symbols=set(S_mapped.values()))
                    dfa_to_check10 = dp.rename_dfa_transitions(DFA.from_nfa(nfa10), S_mapped)
                    if dfa_to_check10.issubset(dfa_decl.complement()):
                        flag = True
                        dev_list.append(reg_to_check10)
                reg_to_check10_2 = f"{g(S1)}*({g(S2-{q})}*{q}{g(S2-{q})}*){g(S1)}*(({g(S2-{q})}*{q}{g(S2-{q})}*){g(S1)}*){{2,2}}"
                nfa10_2 = NFA.from_regex(reg_to_check10_2, input_symbols=set(S_mapped.values()))
                dfa_to_check10_2 = dp.rename_dfa_transitions(DFA.from_nfa(nfa10_2), S_mapped)
                if dfa_to_check10_2.issubset(dfa_decl.complement()):
                    flag = True
                    dev_list.append(reg_to_check10_2)
                reg_to_check10_3 = f"{g(S1)}*({g(S2-{q})}*{q}{g(S2-{q})}*){g(S1)}*(({g(S2-{q})}*{q}{g(S2-{q})}*){g(S1)}*){{3,3}}"
                nfa10_3 = NFA.from_regex(reg_to_check10_3, input_symbols=set(S_mapped.values()))
                dfa_to_check10_3 = dp.rename_dfa_transitions(DFA.from_nfa(nfa10_3), S_mapped)
                if dfa_to_check10_3.issubset(dfa_decl.complement()):
                    flag = True
                    dev_list.append(reg_to_check10_3)




    elif cut_type == 'loop_tau':
        for p in SS1:
            # p = S_mapped[pp]
            reg_to_check11 = f"{g(S1)}*{p}{g(S1)}"
            nfa11 = NFA.from_regex(reg_to_check11, input_symbols=set(S_mapped.values()))
            dfa_to_check11 = dp.rename_dfa_transitions(DFA.from_nfa(nfa11), S_mapped)
            if dfa_to_check11.issubset(dfa_decl.complement()):
                flag = True
                dev_list.append(reg_to_check11)
            for pp in SS1:
                reg_to_check12 = f"({g(S1 - {p})}*{p}{g(S1 - {p})}*)({g(S1 - {pp})}*{pp}{g(S1 - {pp})}*){{1,1}}"
                nfa12 = NFA.from_regex(reg_to_check12, input_symbols=set(S_mapped.values()))
                dfa_to_check12 = dp.rename_dfa_transitions(DFA.from_nfa(nfa12), S_mapped)
                if dfa_to_check12.issubset(dfa_decl.complement()):
                    flag = True
                    dev_list.append(reg_to_check12)
            reg_to_check12_2 = f"({g(S1 - {p})}*{p}{g(S1 - {p})}*)({g(S1 - {p})}*{p}{g(S1 - {p})}*){{2,2}}"
            nfa12_2 = NFA.from_regex(reg_to_check12_2, input_symbols=set(S_mapped.values()))
            dfa_to_check12_2 = dp.rename_dfa_transitions(DFA.from_nfa(nfa12_2), S_mapped)
            if dfa_to_check12_2.issubset(dfa_decl.complement()):
                flag = True
                dev_list.append(reg_to_check12_2)
            reg_to_check12_3 = f"({g(S1 - {p})}*{p}{g(S1 - {p})}*)({g(S1 - {p})}*{p}{g(S1 - {p})}*){{3,3}}"
            nfa12_3 = NFA.from_regex(reg_to_check12_3, input_symbols=set(S_mapped.values()))
            dfa_to_check12_3 = dp.rename_dfa_transitions(DFA.from_nfa(nfa12_3), S_mapped)
            if dfa_to_check12_3.issubset(dfa_decl.complement()):
                flag = True
                dev_list.append(reg_to_check12_3)


    elif cut_type == 'exc_tau':
        for p in SS1:
            reg_to_check13 = f"{g(S1 - {p})}*{p}{g(S1 - {p})}*"
            nfa13 = NFA.from_regex(reg_to_check13, input_symbols=set(S_mapped.values()))
            dfa_to_check13 = dp.rename_dfa_transitions(DFA.from_nfa(nfa13), S_mapped)
            if dfa_to_check13.issubset(dfa_decl.complement()):
                # print(f"p is {p}")
                # print(reg_to_check)
                flag = True
                dev_list.append(reg_to_check13)
                # block = True

        reg_to_check14 = ""
        nfa14 = NFA.from_regex(reg_to_check14, input_symbols=set(S_mapped.values()))
        dfa_to_check14 = dp.rename_dfa_transitions(DFA.from_nfa(nfa14), S_mapped)
        if dfa_to_check14.issubset(dfa_decl.complement()):
            # print(f"p is {p}")
            # print(reg_to_check)
            flag = True
            dev_list.append(reg_to_check14)

    return flag,dev_list


def check_all_single(activity, dfa_decl, cut_type, S_mapped):
    flag = False
    dev_list = []
    if cut_type == 'single_single':
        reg_to_check = f"{activity}"
        nfa = NFA.from_regex(reg_to_check, input_symbols=set(S_mapped.values()))
        dfa_to_check = dp.rename_dfa_transitions(DFA.from_nfa(nfa), S_mapped)
        if dfa_to_check.issubset(dfa_decl.complement()):
            flag = True
            dev_list.append(reg_to_check)
    elif cut_type == 'xor_single':
        reg_to_check = f"{activity}"
        nfa = NFA.from_regex(reg_to_check, input_symbols=set(S_mapped.values()))
        dfa_to_check = dp.rename_dfa_transitions(DFA.from_nfa(nfa), S_mapped)
        if dfa_to_check.issubset(dfa_decl.complement()):
            flag = True
            dev_list.append(reg_to_check)
        reg_to_check2 = ""
        nfa2 = NFA.from_regex(reg_to_check2, input_symbols=set(S_mapped.values()))
        dfa_to_check2 = dp.rename_dfa_transitions(DFA.from_nfa(nfa2), S_mapped)
        if dfa_to_check2.issubset(dfa_decl.complement()):
            # print(f"p is {p}")
            # print(reg_to_check)
            flag = True
            dev_list.append(reg_to_check2)
    elif cut_type == 'loop_single':
        reg_to_check = f"{activity}"
        nfa = NFA.from_regex(reg_to_check, input_symbols=set(S_mapped.values()))
        dfa_to_check = dp.rename_dfa_transitions(DFA.from_nfa(nfa), S_mapped)
        if dfa_to_check.issubset(dfa_decl.complement()):
            flag = True
            dev_list.append(reg_to_check)
        reg_to_check2 = f"{activity}{activity}"
        nfa2 = NFA.from_regex(reg_to_check2, input_symbols=set(S_mapped.values()))
        dfa_to_check2 = dp.rename_dfa_transitions(DFA.from_nfa(nfa2), S_mapped)
        if dfa_to_check2.issubset(dfa_decl.complement()):
            flag = True
            dev_list.append(reg_to_check2)
        reg_to_check3 = f"{activity}{activity}{activity}"
        nfa3 = NFA.from_regex(reg_to_check3, input_symbols=set(S_mapped.values()))
        dfa_to_check3 = dp.rename_dfa_transitions(DFA.from_nfa(nfa3), S_mapped)
        if dfa_to_check3.issubset(dfa_decl.complement()):
            flag = True
            dev_list.append(reg_to_check3)


    return flag,dev_list


templates = dp.templates
S_mapped = {'a':'a','b':'b','c':'c','d':'d'}
# lookup_table_relation = {}
# lookup_table_single = {}
lookup_table = {}
contradictory_example = {}

'''
Scenarios:
first element: 'a' in 'A'
second element: 'b' in 'A'
third element: 'B' is empty
forth element: base cases
'''
scenarios = {"1100":{'A': {'a','b'}, 'B':{'c','d'}}, "0000":{'A':{'c','d'}, 'B':{'a','b'}}, "1000":{'A':{'a','c'}, 'B':{'b','d'}}, "0100":{'A':{'c','b'}, 'B':{'a','d'}},"1110":{'A':{'a','b','c','d'}, 'B':{}},"0001":{'activity':'a'}}
cut_types = ['seq', 'exc', 'par','loop']
cut_types_tau = ['loop_tau', 'exc_tau']
cut_types_base = ['single_single', 'xor_single', 'loop_single']
# templates_used = ["AtMost1", "AtLeast1", "Response", "Precedence", "CoExistence", "NotCoExistence", "NotSuccession", "RespondedExistence"]

single_templates = {}

for t in templates:
    if templates[t][1]==1:
        dfa_decl = dp.gen_reg_dfa(t, ['a'], S_mapped)
        lookup_table[t] = {}
        contradictory_example[t] = {}
        for sc in {"1000","0000"}:
            lookup_table[t][sc]= set()
            contradictory_example[t][sc] = set()
            for ct in cut_types:
                fl, dev_list = check_all(scenarios[sc]["A"], scenarios[sc]["B"],scenarios[sc]["A"], scenarios[sc]["B"], dfa_decl, ct,S_mapped)
                if fl is True:
                    # lookup_table_single[t][sc].add((ct,dev_list[0]))
                    contradictory_example[t][sc].add((ct,dev_list[0]))
                    lookup_table[t][sc].add(ct)
                # print(fl)
                print(f'The result for declare rule {t}(a,b) considering cut type {ct} and scenario {sc} is {fl}')
        sc = "1110"
        lookup_table[t][sc] = set()
        contradictory_example[t][sc] = set()
        for ct in cut_types_tau:
            fl, dev_list = check_all(scenarios[sc]["A"], scenarios[sc]["B"], scenarios[sc]["A"], scenarios[sc]["B"],
                                     dfa_decl, ct, S_mapped)
            if fl is True:
                # lookup_table_single[t][sc].add((ct, dev_list[0]))
                contradictory_example[t][sc].add((ct, dev_list[0]))
                lookup_table[t][sc].add(ct)
            # print(fl)
            print(f'The result for declare rule {t}(a,b) considering cut type {ct} and scenario {sc} is {fl}')
        sc = "0001"
        lookup_table[t][sc] = set()
        contradictory_example[t][sc] = set()
        for ct in cut_types_base:
            fl, dev_list = check_all_single(scenarios[sc]["activity"], dfa_decl, ct, S_mapped)
            if fl is True:
                # lookup_table_single[t][sc].add((ct, dev_list[0]))
                contradictory_example[t][sc].add((ct, dev_list[0]))
                lookup_table[t][sc].add(ct)
            # print(fl)
            print(f'The result for declare rule {t}(a,b) considering cut type {ct} and scenario {sc} is {fl}')


    elif templates[t][1]==2:
        lookup_table[t] = {}
        contradictory_example[t] = {}
        for sc in {"1100", "0000", "1000", "0100"}:
            dfa_decl = dp.gen_reg_dfa(t, ['a', 'b'], S_mapped)
            lookup_table[t][sc]= set()
            contradictory_example[t][sc] = set()
            for ct in cut_types:
                fl, dev_list = check_all(scenarios[sc]["A"], scenarios[sc]["B"],scenarios[sc]["A"], scenarios[sc]["B"], dfa_decl, ct,S_mapped)
                if fl is True:
                    # lookup_table_relation[t][sc].add((ct,dev_list[0]))
                    contradictory_example[t][sc].add((ct, dev_list[0]))
                    lookup_table[t][sc].add(ct)
                # print(fl)
                print(f'The result for declare rule {t}(a,b) considering cut type {ct} and scenario {sc} is {fl}')
        sc = "1110"
        lookup_table[t][sc] = set()
        contradictory_example[t][sc] = set()
        for ct in cut_types_tau:
            fl, dev_list = check_all(scenarios[sc]["A"], scenarios[sc]["B"], scenarios[sc]["A"], scenarios[sc]["B"],
                                     dfa_decl, ct, S_mapped)
            if fl is True:
                # lookup_table_relation[t][sc].add((ct, dev_list[0]))
                contradictory_example[t][sc].add((ct, dev_list[0]))
                lookup_table[t][sc].add(ct)
            # print(fl)
            print(f'The result for declare rule {t}(a,b) considering cut type {ct} and scenario {sc} is {fl}')
    else:
        print(f'Template {t} with {templates[t][1]} input parameters is not supported!')


# for x in lookup_table.keys():
#     print(x)
#     for y in lookup_table[x]:
#         print(f'{x}: {y} ---> {lookup_table[x][y]}')

# for x in lookup_table_relation.keys():
#     print(x)
#     for y in lookup_table_relation[x]:
#         print(f'{x}: {y} ---> {lookup_table_relation[x][y]}')

df = pd.DataFrame(lookup_table).T
df.to_csv('lookup_table.csv',sep=';')

df_detailed = pd.DataFrame(contradictory_example).T
df_detailed.to_csv('lookup_table_detailed.csv',sep=';')