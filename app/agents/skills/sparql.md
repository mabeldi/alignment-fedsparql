---
name: sparql-query-generator
description: Generates grounded, standards-compliant SPARQL 1.1 SELECT DISTINCT queries, including federated SERVICE queries. Use when translating a natural-language question plus supplied RDF schema, paths, endpoints, or mappings into a query, especially when deciding join versus UNION and linking joined cross-endpoint blocks through owl:sameAs.
---

# SPARQL rules

## General
- Standard SPARQL 1.1 only; no vendor extensions.
- Query form: always `SELECT DISTINCT`. Never ASK, CONSTRUCT or DESCRIBE.
- Invent nothing: every class, property, endpoint, mapping and constant comes from the input.
- Add only patterns that bind or constrain the answer. Add FILTER, OPTIONAL, aggregates, ORDER BY, LIMIT, OFFSET only when the question requires them.
- Declare every prefix used (full IRIs also allowed).
- Work order: what to return → patterns that bind/constrain it → SERVICE placement → join vs UNION → owl:sameAs links (joins only) → modifiers → check.

## Joins
- Patterns in one group are all required. They join on shared variables; reuse a variable only for the same RDF term.
- Patterns meant to describe one result must be connected by a variable or a link, else they form a cartesian product.
- `;` only to repeat a subject.

## Join vs UNION
- Join: facts/conditions that must hold in the SAME row.
- UNION: alternatives, each branch alone satisfying the same part of the request. Brace every branch.
- Never replace a required join with UNION.
- SERVICE picks WHERE a pattern runs; a join puts facts in one row; UNION gives alternative rows.
- SERVICE SILENT only when a failing endpoint may count as a missing result.

Alternative instances from two endpoints (UNION: shared variable name, no owl:sameAs):
```
{ SERVICE <A> { ?x a p1:C . } }
UNION
{ SERVICE <B> { ?x a p2:C . } }
```
Complementary facts about one entity (join: one variable per endpoint, linked by owl:sameAs):
```
SERVICE <A> { ?xA a p1:C ; p1:f ?v1 . }
SERVICE <B> { ?xB owl:sameAs ?xA ; p2:g ?v2 . }
```

## owl:sameAs depends on the query shape
- Two endpoints never share an instance IRI.
- JOIN between two SERVICE blocks → required: give the entity one variable per endpoint; the joined block opens with `?own owl:sameAs ?other .` inside its own SERVICE (each endpoint stores links from its own instances).
- UNION between SERVICE blocks → never: branches are independent and share one variable name.

## OPTIONAL, FILTER, negation
- OPTIONAL only if the row must survive when that data is missing.
- FILTER only for requested constraints, placed where its variables are bound.
- FILTER NOT EXISTS / MINUS only for requested negation; choose by variable correlation.

## Paths and aggregates
- Property paths only when they exactly match the supplied path; prefer plain triples when equally correct.
- Projected non-aggregate variables must fit the GROUP BY.

## Check before replying
Valid SPARQL 1.1; `SELECT DISTINCT` form; every prefix declared; braces and parentheses balanced; every projected variable bindable; all vocabulary from the input; required facts joined, alternatives in braced UNION branches, optional facts in OPTIONAL; each pattern in its intended SERVICE; every cross-SERVICE join linked by owl:sameAs and no owl:sameAs between UNION branches; no cartesian product; no unneeded pattern, filter, modifier or output variable.
