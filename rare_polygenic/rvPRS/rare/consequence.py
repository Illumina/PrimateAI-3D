
from typing import Dict, Iterable, Set

from rvPRS.rare.variant import Variant

PTV = set([
    'frameshift_variant', 
    'stop_gained', 
    'splice_donor_variant', 
    'splice_acceptor_variant',
    'start_lost',
    'stop_lost',
    ])

MISSENSE = set([
    'missense_variant',
])

SYNONYMOUS = set([
    'synonymous_variant',
])

def group_by_consequence(variants: Iterable[Variant], spliceai_threshold=0.2) -> Dict[str, Set[Variant]]:
    ''' group variants by consequence type
    '''
    cqs = {'syn': set(), 'del': set(), 'ptv': set()}
    for var in variants:
        if var.consequence in PTV:
            var.missense_pathogenicity = 1.0
            cqs['ptv'].add(var)
            cqs['del'].add(var)
        elif var.consequence in MISSENSE and var.missense_pathogenicity is not None:
            cqs['del'].add(var)
        elif var.spliceai is not None and var.spliceai > spliceai_threshold:
            var.missense_pathogenicity = 1.0
            cqs['ptv'].add(var)
            cqs['del'].add(var)
        elif var.consequence in SYNONYMOUS:
            var['missense_pathogenicity'] = 0.0
            cqs['syn'].add(var)
    return cqs
