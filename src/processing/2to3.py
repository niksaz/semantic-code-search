from lib2to3 import refactor

code = 'print "old-fashioned print"'
fixes = set(refactor.get_fixers_from_package("lib2to3.fixes"))
conversion_tool = refactor.RefactoringTool(fixes)
converted_code = str(conversion_tool.refactor_string(code + '\n', 'some_name'))
print(converted_code)  # print("old-fashioned print")
