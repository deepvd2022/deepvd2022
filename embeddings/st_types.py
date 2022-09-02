
import re
import os
import json

def findAllFile(base, full=True):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if full:
                yield os.path.join(root, f)
            else:
                yield f

def split_trees(tree):
    """
    joern 生成的 AST、PDG 等融合在一块，这个 function 把它们分开。
    """
    res = {}
    tree_type = ""
    tree_body = ""
    for line in tree.splitlines():
        if line.strip() == "":
            continue
        if line[0] == '#':
            if tree_type != "" and tree_body != "":
                res[ tree_type ] = tree_body
                tree_body = ""

            tree_type = line[2:].strip()
            # print(tree_type)
        elif tree_type != "":
            tree_body += line + "\n"
    if tree_body != "":
        res[tree_type] = tree_body
    return res


class Tree(object):
    # Use this structure to create tree data
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.id = None

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth


def get_joern_id(line):
    p1 = re.compile(r'joern_id_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''

def get_joern_type(line):
    p1 = re.compile(r'joern_type_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''

def get_joern_name(line):
    p1 = re.compile(r'joern_name_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''

def get_joern_line_no(line):
    p1 = re.compile(r'joern_line_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''

class StType():
    def __init__(self, str_ast):
        self.st_types = {}
        self.nodes = {}
        self.type_list = ['UNKNOWN', 'METHOD', 'METHOD_PARAMETER_IN', 'METHOD_RETURN', 'BLOCK', 'CALL', 'IDENTIFIER',
                          'CONTROL_STRUCTURE', 'RETURN', 'METHOD_PARAMETER_OUT', 'LITERAL', 'FIELD_IDENTIFIER']
        self.appeared_types = []
        self.do(str_ast)

    def find_root(self, edges):
        appeared = []
        for edge in edges.keys():
            for node in edges[edge]:
                if node not in appeared:
                    appeared.append(node)
        for key in edges.keys():
            if key not in appeared:
                return key
        return None

    def build_tree(self, edge_list, id):
        root = Tree()
        root.id = id
        if edge_list.get(id):
            for child_id in edge_list.get(id):
                new_child = self.build_tree(edge_list, child_id)
                root.add_child(new_child)
        return root

    def walk_tree(self, tree):
        line_no = int(self.nodes[ tree.id ]['line_no'])
        cur_type = self.nodes[ tree.id ]['st_type']

        if cur_type not in self.appeared_types:
            self.appeared_types.append(cur_type)

        if cur_type in self.type_list:
            cur_type_id = self.type_list.index(cur_type)
        else:
            cur_type_id = 0

        if line_no not in self.st_types.keys():
            self.st_types[line_no] = cur_type_id

        for child in tree.children:
            self.walk_tree(child)

    def do(self, str_ast):
        children = {}

        for line in str_ast.splitlines():
            if line.find('" -->> "') > -1:
                a, b = line.split('" -->> "', 1)

                id1 = get_joern_id(a)
                id2 = get_joern_id(b)

                no1 = get_joern_line_no(a)
                no2 = get_joern_line_no(b)

                t1 = get_joern_type(a)
                t2 = get_joern_type(b)

                if id1 not in self.nodes.keys():
                    self.nodes[id1] = {
                        'id': id1,
                        'line_no': no1,
                        'st_type': t1
                    }

                if id2 not in self.nodes.keys():
                    self.nodes[id2] = {
                        'id': id2,
                        'line_no': no2,
                        'st_type': t2
                    }

                if id1 not in children.keys():
                    children[id1] = []

                children[id1].append(id2)
        root_id = self.find_root(children)
        if root_id is None:
            return
        tree = self.build_tree(children, root_id)

        # 遍历 tree，对于每一个 statement，因为它有多个 token，所以它可能在 AST 中出现多次，取其在 AST 中最上面的节点的 joern type。
        self.walk_tree(tree)

    def get_st_types(self):
        return self.st_types


if __name__ == '__main__':
    str_ast = '''{
      edge[color=green3,constraint=true]
      "joern_id_(80)_joern_code_(return 0;)_joern_type_(RETURN)_joern_name_joern_line_(12)" -->> "joern_id_(79)_joern_code_(0)_joern_type_(LITERAL)_joern_name_joern_line_(12)" 
       "joern_id_(82)_joern_code_(return 1;)_joern_type_(RETURN)_joern_name_joern_line_(10)" -->> "joern_id_(81)_joern_code_(1)_joern_type_(LITERAL)_joern_name_joern_line_(10)" 
       "joern_id_(83)_joern_code_()_joern_type_(BLOCK)_joern_name_joern_line_(9)" -->> "joern_id_(82)_joern_code_(return 1;)_joern_type_(RETURN)_joern_name_joern_line_(10)" 
       "joern_id_(87)_joern_code_(dump_dir_name + len)_joern_type_(CALL)_joern_name_(<operator>.addition)_joern_line_(8)" -->> "joern_id_(86)_joern_code_(dump_dir_name)_joern_type_(IDENTIFIER)_joern_name_(dump_dir_name)_joern_line_(8)" 
       "joern_id_(87)_joern_code_(dump_dir_name + len)_joern_type_(CALL)_joern_name_(<operator>.addition)_joern_line_(8)" -->> "joern_id_(85)_joern_code_(len)_joern_type_(IDENTIFIER)_joern_name_(len)_joern_line_(8)" 
       "joern_id_(88)_joern_code_(strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(strstr)_joern_line_(8)" -->> "joern_id_(87)_joern_code_(dump_dir_name + len)_joern_type_(CALL)_joern_name_(<operator>.addition)_joern_line_(8)" 
       "joern_id_(88)_joern_code_(strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(strstr)_joern_line_(8)" -->> "joern_id_(84)_joern_code_(/.)_joern_type_(LITERAL)_joern_name_joern_line_(8)" 
       "joern_id_(89)_joern_code_(!strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(<operator>.logicalNot)_joern_line_(8)" -->> "joern_id_(88)_joern_code_(strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(strstr)_joern_line_(8)" 
       "joern_id_(93)_joern_code_(dump_dir_name[len])_joern_type_(CALL)_joern_name_(<operator>.indirectIndexAccess)_joern_line_(6)" -->> "joern_id_(92)_joern_code_(dump_dir_name)_joern_type_(IDENTIFIER)_joern_name_(dump_dir_name)_joern_line_(6)" 
       "joern_id_(93)_joern_code_(dump_dir_name[len])_joern_type_(CALL)_joern_name_(<operator>.indirectIndexAccess)_joern_line_(6)" -->> "joern_id_(91)_joern_code_(len)_joern_type_(IDENTIFIER)_joern_name_(len)_joern_line_(6)" 
       "joern_id_(94)_joern_code_(dump_dir_name[len] == /)_joern_type_(CALL)_joern_name_(<operator>.equals)_joern_line_(6)" -->> "joern_id_(93)_joern_code_(dump_dir_name[len])_joern_type_(CALL)_joern_name_(<operator>.indirectIndexAccess)_joern_line_(6)" 
       "joern_id_(94)_joern_code_(dump_dir_name[len] == /)_joern_type_(CALL)_joern_name_(<operator>.equals)_joern_line_(6)" -->> "joern_id_(90)_joern_code_(/)_joern_type_(LITERAL)_joern_name_joern_line_(6)" 
       "joern_id_(95)_joern_code_(dump_dir_name[len] == /n    /* must not contain /. anywhere (IOW: disallow .. component) */n     && !strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(<operator>.logicalAnd)_joern_line_(6)" -->> "joern_id_(94)_joern_code_(dump_dir_name[len] == /)_joern_type_(CALL)_joern_name_(<operator>.equals)_joern_line_(6)" 
       "joern_id_(95)_joern_code_(dump_dir_name[len] == /n    /* must not contain /. anywhere (IOW: disallow .. component) */n     && !strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(<operator>.logicalAnd)_joern_line_(6)" -->> "joern_id_(89)_joern_code_(!strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(<operator>.logicalNot)_joern_line_(8)" 
       "joern_id_(100)_joern_code_(strncmp(dump_dir_name, g_settings_dump_location, len))_joern_type_(CALL)_joern_name_(strncmp)_joern_line_(5)" -->> "joern_id_(99)_joern_code_(dump_dir_name)_joern_type_(IDENTIFIER)_joern_name_(dump_dir_name)_joern_line_(5)" 
       "joern_id_(100)_joern_code_(strncmp(dump_dir_name, g_settings_dump_location, len))_joern_type_(CALL)_joern_name_(strncmp)_joern_line_(5)" -->> "joern_id_(98)_joern_code_(g_settings_dump_location)_joern_type_(IDENTIFIER)_joern_name_(g_settings_dump_location)_joern_line_(5)" 
       "joern_id_(100)_joern_code_(strncmp(dump_dir_name, g_settings_dump_location, len))_joern_type_(CALL)_joern_name_(strncmp)_joern_line_(5)" -->> "joern_id_(97)_joern_code_(len)_joern_type_(IDENTIFIER)_joern_name_(len)_joern_line_(5)" 
       "joern_id_(101)_joern_code_(strncmp(dump_dir_name, g_settings_dump_location, len) == 0)_joern_type_(CALL)_joern_name_(<operator>.equals)_joern_line_(5)" -->> "joern_id_(100)_joern_code_(strncmp(dump_dir_name, g_settings_dump_location, len))_joern_type_(CALL)_joern_name_(strncmp)_joern_line_(5)" 
       "joern_id_(101)_joern_code_(strncmp(dump_dir_name, g_settings_dump_location, len) == 0)_joern_type_(CALL)_joern_name_(<operator>.equals)_joern_line_(5)" -->> "joern_id_(96)_joern_code_(0)_joern_type_(LITERAL)_joern_name_joern_line_(5)" 
       "joern_id_(102)_joern_code_(strncmp(dump_dir_name, g_settings_dump_location, len) == 0n     && dump_dir_name[len] == /n    /* must not contain /. anywhere (IOW: disallow .. component) */n     && !strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(<operator>.logicalAnd)_joern_line_(5)" -->> "joern_id_(101)_joern_code_(strncmp(dump_dir_name, g_settings_dump_location, len) == 0)_joern_type_(CALL)_joern_name_(<operator>.equals)_joern_line_(5)" 
       "joern_id_(102)_joern_code_(strncmp(dump_dir_name, g_settings_dump_location, len) == 0n     && dump_dir_name[len] == /n    /* must not contain /. anywhere (IOW: disallow .. component) */n     && !strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(<operator>.logicalAnd)_joern_line_(5)" -->> "joern_id_(95)_joern_code_(dump_dir_name[len] == /n    /* must not contain /. anywhere (IOW: disallow .. component) */n     && !strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(<operator>.logicalAnd)_joern_line_(6)" 
       "joern_id_(103)_joern_code_(if (strncmp(dump_dir_name, g_settings_dump_location, len) == 0n     && dump_dir_name[len] == /n    /* must not contain /. anywhere (IOW: disallow .. component) */n     && !strstr(dump_dir_name + len, /.)n    ))_joern_type_(CONTROL_STRUCTURE)_joern_name_joern_line_(5)" -->> "joern_id_(102)_joern_code_(strncmp(dump_dir_name, g_settings_dump_location, len) == 0n     && dump_dir_name[len] == /n    /* must not contain /. anywhere (IOW: disallow .. component) */n     && !strstr(dump_dir_name + len, /.))_joern_type_(CALL)_joern_name_(<operator>.logicalAnd)_joern_line_(5)" 
       "joern_id_(103)_joern_code_(if (strncmp(dump_dir_name, g_settings_dump_location, len) == 0n     && dump_dir_name[len] == /n    /* must not contain /. anywhere (IOW: disallow .. component) */n     && !strstr(dump_dir_name + len, /.)n    ))_joern_type_(CONTROL_STRUCTURE)_joern_name_joern_line_(5)" -->> "joern_id_(83)_joern_code_()_joern_type_(BLOCK)_joern_name_joern_line_(9)" 
       "joern_id_(105)_joern_code_(strlen(g_settings_dump_location))_joern_type_(CALL)_joern_name_(strlen)_joern_line_(3)" -->> "joern_id_(104)_joern_code_(g_settings_dump_location)_joern_type_(IDENTIFIER)_joern_name_(g_settings_dump_location)_joern_line_(3)" 
       "joern_id_(107)_joern_code_(len = strlen(g_settings_dump_location))_joern_type_(CALL)_joern_name_(<operator>.assignment)_joern_line_(3)" -->> "joern_id_(106)_joern_code_(len)_joern_type_(IDENTIFIER)_joern_name_(len)_joern_line_(3)" 
       "joern_id_(107)_joern_code_(len = strlen(g_settings_dump_location))_joern_type_(CALL)_joern_name_(<operator>.assignment)_joern_line_(3)" -->> "joern_id_(105)_joern_code_(strlen(g_settings_dump_location))_joern_type_(CALL)_joern_name_(strlen)_joern_line_(3)" 
       "joern_id_(109)_joern_code_()_joern_type_(BLOCK)_joern_name_joern_line_(2)" -->> "joern_id_(107)_joern_code_(len = strlen(g_settings_dump_location))_joern_type_(CALL)_joern_name_(<operator>.assignment)_joern_line_(3)" 
       "joern_id_(109)_joern_code_()_joern_type_(BLOCK)_joern_name_joern_line_(2)" -->> "joern_id_(103)_joern_code_(if (strncmp(dump_dir_name, g_settings_dump_location, len) == 0n     && dump_dir_name[len] == /n    /* must not contain /. anywhere (IOW: disallow .. component) */n     && !strstr(dump_dir_name + len, /.)n    ))_joern_type_(CONTROL_STRUCTURE)_joern_name_joern_line_(5)" 
       "joern_id_(109)_joern_code_()_joern_type_(BLOCK)_joern_name_joern_line_(2)" -->> "joern_id_(80)_joern_code_(return 0;)_joern_type_(RETURN)_joern_name_joern_line_(12)" 
       "joern_id_(112)_joern_code_joern_type_(METHOD)_joern_name_(dir_is_in_dump_location)_joern_line_(1)" -->> "joern_id_(111)_joern_code_(const char *dump_dir_name)_joern_type_(METHOD_PARAMETER_IN)_joern_name_(dump_dir_name)_joern_line_(1)" 
       "joern_id_(112)_joern_code_joern_type_(METHOD)_joern_name_(dir_is_in_dump_location)_joern_line_(1)" -->> "joern_id_(110)_joern_code_(RET)_joern_type_(METHOD_RETURN)_joern_name_joern_line_(1)" 
       "joern_id_(112)_joern_code_joern_type_(METHOD)_joern_name_(dir_is_in_dump_location)_joern_line_(1)" -->> "joern_id_(109)_joern_code_()_joern_type_(BLOCK)_joern_name_joern_line_(2)" 
       "joern_id_(112)_joern_code_joern_type_(METHOD)_joern_name_(dir_is_in_dump_location)_joern_line_(1)" -->> "joern_id_(4)_joern_code_(const char *dump_dir_name)_joern_type_(METHOD_PARAMETER_OUT)_joern_name_(dump_dir_name)_joern_line_(1)" 
     }
    '''

    st = StType(str_ast)
    st_types = st.get_st_types()
    for k, v in st_types.items():
        print(k, v)

    INPUT_PATH = "data/verstehen"
    OUTPUT_PATH = "data/DeepVD"

    print("INPUT_PATH: {}".format(INPUT_PATH))
    print("OUTPUT_PATH: {}".format(OUTPUT_PATH))

    all_types = []

    ii = 0
    for file in findAllFile(INPUT_PATH, True):

        if not file.endswith("entities_1hop.json"):
            continue

        if ii > 1000:
            break

        edges_file = file.replace("entities_1hop", "edges_1hop")

        if not os.path.exists(edges_file):
            continue

        if ii % 10 == 0:
            print("now: {}".format(ii))
        with open(file) as f:
            entities = json.loads(f.read())
        with open(edges_file) as f:
            rel = json.loads(f.read())

        # C -> D -> F -> A -> B
        for lv0_func_key in rel.keys():

            if lv0_func_key not in entities.keys():
                continue

            lv0_func = entities[lv0_func_key]

            if lv0_func['contents'].strip() == 0:
                continue

            if 'tree' not in lv0_func.keys() or lv0_func['tree'].strip() == '':
                continue

            trees = split_trees(lv0_func['tree'])

            if 'AST' not in trees.keys():
                print("== no AST: {}".format(lv0_func['tree']))
                continue

            st = StType(trees['AST'])
            for t in st.appeared_types:
                if t not in all_types:
                    print("=== new: {}".format(t))
                    all_types.append(t)
        ii += 1
    print(all_types)