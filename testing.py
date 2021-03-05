import ast
thing = ["['bank of america']", '[]', '[]', "['technopark kollam']"]
for org_list in thing:
    print(ast.literal_eval(org_list))
# print((thing))
