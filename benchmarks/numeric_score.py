from decimal import Decimal,InvalidOperation
import re


def numeric(value):
    if value is None:return None
    value=value.strip().replace(' ','')
    if ',' in value and '.' in value:
        decimal=',' if value.rfind(',')>value.rfind('.') else '.'
        value=value.replace('.' if decimal==',' else ',','').replace(decimal,'.')
    elif re.fullmatch(r'-?\d{1,3}(?:[.,]\d{3})+',value):value=value.replace(',','').replace('.','')
    else:value=value.replace(',','.')
    try:return Decimal(value)
    except InvalidOperation:return None
