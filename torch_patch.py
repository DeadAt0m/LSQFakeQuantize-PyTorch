from pathlib import Path
import re
import sys
from copy import deepcopy as copy
import subprocess
import torch

torch_ver = torch.__version__
torch_ver = torch_ver.split('+')[0] if '+' in torch_ver else torch_ver

python_exec = sys.executable
torch_path = Path(torch.__file__).parent.resolve()

find_str_1 = '''
struct ArgumentDef final {
  using GetTypeFn = TypePtr();
  GetTypeFn* getTypeFn;
};
'''

patch_str_1 = '''
struct ArgumentDef final {
  using GetTypeFn = TypePtr();
  GetTypeFn* getTypeFn;
  constexpr ArgumentDef(): getTypeFn(nullptr) {}
  explicit constexpr ArgumentDef(GetTypeFn *getTypeFn): getTypeFn(getTypeFn) {}
};
'''

find_str_2 = r"std::array<ArgumentDef, sizeof...(Ts)>{{ArgumentDef{&getTypePtr_<std::decay_t<Ts>>::call}...}}"
patch_str_2 = r"std::array<ArgumentDef, sizeof...(Ts)>{ArgumentDef(&getTypePtr_<std::decay_t<Ts>>::call)...}"



def patch_torch_infer_schema_h():
    infer_schema_header = torch_path / 'include' / 'ATen' / 'core' / 'op_registration' / 'infer_schema.h'
    if not infer_schema_header.exists():
        print(f'{str(infer_schema_header)} not found')
        return False
    content = infer_schema_header.read_text()
    orig_content = copy(content)
    ret = True
    content = content.replace(find_str_1, patch_str_1)
    ret *= (content.find(find_str_1) == -1)
    content = content.replace(find_str_2, patch_str_2)
    ret *= (content.find(find_str_2) == -1)
    if content != orig_content:
        print(f'Try writing into file: {str(infer_schema_header)}...')
        try:
            infer_schema_header.unlink()
            infer_schema_header.write_text(content)
        except:
            print('You need to execute this as root for proper patching!')
            subprocess.call(['sudo', python_exec, *sys.argv])
            sys.exit()
        print('Success!')
    return ret

if __name__ == '__main__':
    print('Patch infer_schema.h:', patch_torch_infer_schema_h())

    