from llvmlite.binding import load_library_permanently, address_of_symbol

load_library_permanently('./isin.so')

printf_address = address_of_symbol("isin")
print(printf_address)
