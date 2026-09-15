"""
DRAWINGS AGENT v3 — Pipeline completa per analisi disegni tecnici (async)

Pipeline:
  1. File in ingresso → parser specializzato per formato
  2. Estrazione strutturata → JSON con entità, layer, dimensioni, materiali
  3. Render preview → PNG cache (DXF via ezdxf+matplotlib)
  4. Conversione → Markdown strutturato + chunking → ChromaDB
  5. Schede semantiche lazy (via semantic_analyzer)
  6. Cross-referencing → cerca correlazioni in collection documents/financial
  7. LLM risposta streaming con contesto multi-fonte

Formati supportati: DXF, DWG (→DXF), STP/STEP, IFC, STL, SVG, PDF tecnici.

Le funzioni di parsing/rendering sono sync (CPU-bound + I/O locale).
Le chiamate LLM, ChromaDB e cross-ref sono async.

Zero hardcoded: nessuna assunzione su contenuto/lingua/dominio.
"""

import sys
import os
import re
import json
import hashlib
import asyncio
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    CHROMA_PATHS, LLM_MODEL_MAIN, LLM_MODEL_FAST,
    CHUNK_SIZE, CHUNK_OVERLAP, EXTENSIONS,
    MEMORY_PATH, MARKDOWN_CACHE_DIR,
)

import chromadb
import semantic_analyzer as analyzer
from embeddings import open_collection
from llm_client import chat_complete, chat_complete_stream, chat_complete_json

# ── ChromaDB ──────────────────────────────────────────────────────────────────

client     = chromadb.PersistentClient(path=CHROMA_PATHS["drawings"])
collection = open_collection(client, "drawings")

# ── Paths ─────────────────────────────────────────────────────────────────────

MEMORY_FILE    = Path(MEMORY_PATH) / "drawings_memory.json"
MARKDOWN_CACHE = Path(MARKDOWN_CACHE_DIR)
PREVIEW_CACHE  = Path(MARKDOWN_CACHE_DIR) / "previews"

# ── Models ────────────────────────────────────────────────────────────────────

ROUTING_MODEL = LLM_MODEL_FAST
ANSWER_MODEL  = LLM_MODEL_MAIN


# ==============================================================================
#  FASE 2 — PARSER SPECIALIZZATI (sync, CPU-bound)
# ==============================================================================

# ── DXF Parser (ezdxf) ───────────────────────────────────────────────────────

def parse_dxf(filepath) -> dict:
    """
    Parser completo per file DXF via ezdxf.

    Estrae:
      - Metadata: versione, unità, bounding box
      - Layer con conteggio entità per tipo
      - Testi e annotazioni (TEXT, MTEXT)
      - Quote dimensionali (DIMENSION) con valori misurati
      - Blocchi/inserimenti con attributi (INSERT + ATTRIB)
      - Statistiche geometriche per tipo di entità
    """
    try:
        import ezdxf
        from ezdxf import bbox as ezdxf_bbox
    except ImportError:
        return {"error": "ezdxf non installato", "filename": Path(filepath).name}

    p = Path(filepath)
    result = {
        "filename": p.name,
        "format": "DXF",
        "filepath": str(filepath),
        "filesize_kb": round(p.stat().st_size / 1024, 1),
    }

    try:
        doc = ezdxf.readfile(str(filepath))
    except Exception as e:
        result["error"] = f"Errore lettura: {e}"
        return result

    # Metadata
    result["dxf_version"] = doc.dxfversion
    units_map = {
        0: "Senza unità", 1: "Pollici", 2: "Piedi", 3: "Miglia",
        4: "Millimetri", 5: "Centimetri", 6: "Metri", 7: "Chilometri",
    }
    try:
        unit_code = doc.header.get("$INSUNITS", 0)
        result["units"] = units_map.get(unit_code, f"Codice {unit_code}")
        result["units_code"] = unit_code
    except Exception:
        result["units"] = "Non specificata"

    # Layer
    layers_info = {}
    for layer in doc.layers:
        layers_info[layer.dxf.name] = {
            "color": layer.dxf.color,
            "is_on": layer.is_on(),
            "is_frozen": layer.is_frozen(),
        }
    result["layers"] = layers_info

    # Itera modelspace
    msp = doc.modelspace()

    entity_counts = {}
    texts = []
    dimensions = []
    blocks = {}
    block_attribs = {}

    for entity in msp:
        etype = entity.dxftype()
        entity_counts[etype] = entity_counts.get(etype, 0) + 1

        # ── Testi
        if etype == "TEXT":
            try:
                t = entity.dxf.text.strip()
                if t:
                    texts.append(t)
            except Exception:
                pass

        elif etype == "MTEXT":
            try:
                t = entity.plain_mtext().strip()
                if t:
                    texts.append(t)
            except Exception:
                pass

        # ── Dimensioni
        elif etype == "DIMENSION":
            dim_info = {}
            try:
                dim_info["text"] = entity.dxf.text or ""
            except Exception:
                dim_info["text"] = ""

            # Tenta di leggere il valore misurato reale
            try:
                measurement = entity.get_measurement()
                if measurement is not None:
                    dim_info["measurement"] = round(measurement, 4)
            except Exception:
                pass

            # Tipo di dimensione
            try:
                dim_info["type"] = entity.dxf.dimtype
            except Exception:
                pass

            dimensions.append(dim_info)

        # ── Blocchi / Inserimenti
        elif etype == "INSERT":
            try:
                block_name = entity.dxf.name
                blocks[block_name] = blocks.get(block_name, 0) + 1

                # Estrai attributi del blocco
                attribs = {}
                for attrib in entity.attribs:
                    try:
                        tag = attrib.dxf.tag
                        val = attrib.dxf.text
                        if tag and val:
                            attribs[tag] = val
                    except Exception:
                        pass
                if attribs and block_name not in block_attribs:
                    block_attribs[block_name] = attribs
            except Exception:
                pass

    result["entity_counts"] = entity_counts
    result["texts"] = texts[:200]  # cap per evitare JSON enormi
    result["dimensions"] = dimensions[:100]
    result["blocks"] = blocks
    result["block_attributes"] = block_attribs

    # Bounding box
    try:
        cache = ezdxf_bbox.Cache()
        box = ezdxf_bbox.extents(msp, cache=cache)
        if box.has_data:
            result["bounding_box"] = {
                "min": [round(v, 4) for v in box.extmin],
                "max": [round(v, 4) for v in box.extmax],
                "size": [round(box.extmax[i] - box.extmin[i], 4) for i in range(3)],
            }
    except Exception:
        pass

    # Conteggio per layer
    layer_entity_counts = {}
    try:
        from ezdxf.groupby import groupby
        groups = groupby(entities=msp, dxfattrib="layer")
        for layer_name, entities_in_layer in groups.items():
            layer_entity_counts[layer_name] = len(list(entities_in_layer))
    except Exception:
        pass
    if layer_entity_counts:
        result["entities_per_layer"] = layer_entity_counts

    return result


# ── STEP/STP Parser ───────────────────────────────────────────────────────────

def parse_stp(filepath) -> dict:
    """
    Parser per file STEP/STP.

    Strategia:
      - Parsing testuale del formato ISO 10303: regex su entità STEP
      - Estrae: prodotti/componenti, superfici, solidi, materiali, unità
      - Analisi geometrica: rapporto cilindriche/planari per inferire tipo pezzo
    """
    p = Path(filepath)
    result = {
        "filename": p.name,
        "format": "STEP",
        "filepath": str(filepath),
        "filesize_kb": round(p.stat().st_size / 1024, 1),
    }

    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        result["error"] = f"Errore lettura: {e}"
        return result

    # Metadata STEP
    desc = re.findall(r"FILE_DESCRIPTION\s*\(\s*\((.*?)\)", content)
    if desc:
        result["description"] = desc[0].replace("'", "").strip()

    schema = re.findall(r"FILE_SCHEMA\s*\(\s*\((.*?)\)\s*\)", content)
    if schema:
        result["schema"] = schema[0].replace("'", "").strip()

    fname = re.findall(r"FILE_NAME\s*\(\s*'([^']*)'", content)
    if fname:
        result["step_filename"] = fname[0]

    # Unità di misura
    if "MILLIMETRE" in content or "MILLIMETER" in content:
        result["units"] = "Millimetri"
    elif "METRE" in content or "METER" in content:
        result["units"] = "Metri"
    elif "INCH" in content:
        result["units"] = "Pollici"

    # Conteggio entità geometriche
    entity_types = {
        "PRODUCT":               "Prodotti/Componenti",
        "MANIFOLD_SOLID_BREP":   "Solidi B-rep",
        "ADVANCED_FACE":         "Facce",
        "EDGE_CURVE":            "Curve di bordo",
        "CYLINDRICAL_SURFACE":   "Superfici cilindriche",
        "CONICAL_SURFACE":       "Superfici coniche",
        "SPHERICAL_SURFACE":     "Superfici sferiche",
        "TOROIDAL_SURFACE":      "Superfici toroidali",
        "PLANE":                 "Piani",
        "B_SPLINE_SURFACE_WITH_KNOTS": "Superfici B-spline",
        "AXIS2_PLACEMENT_3D":    "Sistemi di riferimento 3D",
        "CIRCLE":                "Cerchi",
        "LINE":                  "Linee",
    }

    counts = {}
    for code, name in entity_types.items():
        # Conta "ENTITY(" e "ENTITY ("
        n = content.count(code + "(") + content.count(code + " (")
        if n > 0:
            counts[name] = n
    result["entity_counts"] = counts

    # Nomi prodotti/componenti
    products = re.findall(r"PRODUCT\s*\(\s*'([^']*)'", content)
    if products:
        unique = list(dict.fromkeys(p.strip() for p in products if p.strip()))
        result["products"] = unique[:50]

    # Nomi parti specifici
    shape_names = re.findall(
        r"SHAPE_DEFINITION_REPRESENTATION\s*\(\s*#\d+\s*,\s*#\d+\s*\)",
        content
    )
    result["shape_definitions"] = len(shape_names) if shape_names else 0

    # Materiali
    materials = re.findall(r"MATERIAL\s*\(\s*'([^']*)'", content)
    if materials:
        result["materials"] = list(dict.fromkeys(m.strip() for m in materials if m.strip()))[:30]

    # Analisi geometrica
    solids = counts.get("Solidi B-rep", 0)
    faces = counts.get("Facce", 0)
    cyl_surf = counts.get("Superfici cilindriche", 0)
    planes = counts.get("Piani", 0)

    analysis = {}
    if solids > 0:
        analysis["distinct_solids"] = solids
    if faces > 0 and solids > 0:
        analysis["avg_faces_per_solid"] = round(faces / solids, 1)
    if faces > 0:
        if cyl_surf > 0:
            ratio = round(cyl_surf / faces * 100, 1)
            analysis["cylindrical_ratio_pct"] = ratio
            if ratio > 30:
                analysis["likely_type"] = "Parte rotazionale/tornita"
        if planes > 0:
            ratio = round(planes / faces * 100, 1)
            analysis["planar_ratio_pct"] = ratio
            if ratio > 50 and cyl_surf / max(faces, 1) < 0.15:
                analysis["likely_type"] = "Parte prismatica/strutturale"
    if analysis:
        result["geometric_analysis"] = analysis

    result["total_lines"] = content.count("\n")
    return result


# ── IFC Parser ────────────────────────────────────────────────────────────────

def parse_ifc(filepath) -> dict:
    """
    Parser per file IFC (BIM) via ifcopenshell.

    Estrae: progetto, piani, elementi edilizi (muri, solette, colonne, porte,
    finestre, scale), materiali, spazi, proprietà (IfcPropertySet).
    """
    p = Path(filepath)
    result = {
        "filename": p.name,
        "format": "IFC",
        "filepath": str(filepath),
        "filesize_kb": round(p.stat().st_size / 1024, 1),
    }

    try:
        import ifcopenshell
    except ImportError:
        result["error"] = "ifcopenshell non installato"
        return result

    try:
        ifc = ifcopenshell.open(str(filepath))
    except Exception as e:
        result["error"] = f"Errore lettura: {e}"
        return result

    result["schema"] = ifc.schema

    # Progetto
    projects = ifc.by_type("IfcProject")
    if projects:
        proj = projects[0]
        result["project_name"] = proj.Name or "N/A"
        if proj.Description:
            result["project_description"] = proj.Description

    # Sito e edificio
    sites = ifc.by_type("IfcSite")
    if sites:
        result["sites"] = [s.Name or "Senza nome" for s in sites]

    buildings = ifc.by_type("IfcBuilding")
    if buildings:
        result["buildings"] = [b.Name or "Senza nome" for b in buildings]

    # Piani
    floors = ifc.by_type("IfcBuildingStorey")
    if floors:
        floor_list = []
        for floor in floors:
            fi = {"name": floor.Name or "Senza nome"}
            try:
                if floor.Elevation is not None:
                    fi["elevation_m"] = round(float(floor.Elevation), 2)
            except Exception:
                pass
            floor_list.append(fi)
        result["floors"] = floor_list

    # Elementi edilizi
    element_types = [
        ("IfcWall", "Pareti"), ("IfcSlab", "Solette"), ("IfcColumn", "Colonne"),
        ("IfcBeam", "Travi"), ("IfcDoor", "Porte"), ("IfcWindow", "Finestre"),
        ("IfcStair", "Scale"), ("IfcRoof", "Tetti"), ("IfcSpace", "Spazi"),
        ("IfcFurniture", "Arredi"), ("IfcCurtainWall", "Pareti vetrate"),
        ("IfcPlate", "Lastre"), ("IfcRailing", "Ringhiere"),
    ]

    elements = {}
    for ifc_type, name in element_types:
        els = ifc.by_type(ifc_type)
        if els:
            elements[name] = {
                "count": len(els),
                "names": list(dict.fromkeys(
                    e.Name or ifc_type for e in els
                ))[:20],
            }
    result["elements"] = elements

    # Materiali
    materials = ifc.by_type("IfcMaterial")
    if materials:
        result["materials"] = list(dict.fromkeys(m.Name for m in materials if m.Name))[:40]

    # Proprietà (IfcPropertySet) — campione
    psets = ifc.by_type("IfcPropertySet")
    if psets:
        result["property_set_count"] = len(psets)
        sample_props = {}
        for pset in psets[:10]:
            props = {}
            try:
                for prop in pset.HasProperties:
                    if hasattr(prop, "NominalValue") and prop.NominalValue:
                        props[prop.Name] = str(prop.NominalValue.wrappedValue)
                if props:
                    sample_props[pset.Name] = props
            except Exception:
                pass
        if sample_props:
            result["sample_properties"] = sample_props

    return result


# ── STL Parser ────────────────────────────────────────────────────────────────

def parse_stl(filepath) -> dict:
    """
    Parser per file STL via trimesh.

    Estrae: triangoli, vertici, bounding box, volume, area superficiale,
    centro di massa, inerzia, watertight check.
    """
    p = Path(filepath)
    result = {
        "filename": p.name,
        "format": "STL",
        "filepath": str(filepath),
        "filesize_kb": round(p.stat().st_size / 1024, 1),
    }

    try:
        import trimesh
    except ImportError:
        # Fallback a numpy-stl
        try:
            from stl import mesh as stl_mesh
            m = stl_mesh.Mesh.from_file(str(filepath))
            result["triangles"] = len(m.vectors)
            min_c = m.vectors.min(axis=(0, 1))
            max_c = m.vectors.max(axis=(0, 1))
            dims = max_c - min_c
            result["bounding_box"] = {
                "size": [round(float(dims[i]), 4) for i in range(3)],
            }
            try:
                vol, _, _ = m.get_mass_properties()
                result["volume"] = round(float(vol), 4)
            except Exception:
                pass
            return result
        except ImportError:
            result["error"] = "Né trimesh né numpy-stl installati"
            return result

    try:
        mesh = trimesh.load(str(filepath), process=True)

        if isinstance(mesh, trimesh.Scene):
            # Se è una scena, prendi la mesh combinata
            mesh = mesh.to_geometry() if hasattr(mesh, "to_geometry") else \
                   trimesh.util.concatenate(mesh.dump())

        result["triangles"] = len(mesh.faces)
        result["vertices"] = len(mesh.vertices)
        result["is_watertight"] = bool(mesh.is_watertight)

        # Bounding box
        bb = mesh.bounding_box
        result["bounding_box"] = {
            "extents": [round(float(v), 4) for v in bb.primitive.extents],
        }

        if mesh.is_watertight:
            result["volume"] = round(float(mesh.volume), 6)
            result["center_mass"] = [round(float(v), 4) for v in mesh.center_mass]

        result["surface_area"] = round(float(mesh.area), 4)
        result["euler_number"] = int(mesh.euler_number)

        # Convexity ratio (volume / convex hull volume)
        try:
            convex_vol = mesh.convex_hull.volume
            if convex_vol > 0:
                result["convexity_ratio"] = round(float(mesh.volume / convex_vol), 4)
        except Exception:
            pass

        # Numero di corpi separati
        try:
            bodies = mesh.split()
            result["separate_bodies"] = len(bodies)
        except Exception:
            pass

    except Exception as e:
        result["error"] = f"Errore lettura: {e}"

    return result


# ── SVG Parser ────────────────────────────────────────────────────────────────

def parse_svg(filepath) -> dict:
    """Parser per file SVG: dimensioni, viewBox, testi, layer/gruppi, path."""
    p = Path(filepath)
    result = {
        "filename": p.name,
        "format": "SVG",
        "filepath": str(filepath),
        "filesize_kb": round(p.stat().st_size / 1024, 1),
    }

    try:
        from lxml import etree
    except ImportError:
        import xml.etree.ElementTree as etree

    try:
        tree = etree.parse(str(filepath))
        root = tree.getroot()

        # Namespace SVG
        ns = {"svg": "http://www.w3.org/2000/svg"}

        result["width"] = root.get("width", "N/A")
        result["height"] = root.get("height", "N/A")
        result["viewBox"] = root.get("viewBox", "N/A")

        # Testi
        texts = []
        for t in root.iter("{http://www.w3.org/2000/svg}text"):
            full_text = "".join(t.itertext()).strip()
            if full_text:
                texts.append(full_text)
        result["texts"] = texts[:100]

        # Layer / Gruppi
        groups = []
        for g in root.iter("{http://www.w3.org/2000/svg}g"):
            gid = g.get("id", "")
            label = g.get("{http://www.inkscape.org/namespaces/inkscape}label", "")
            if gid or label:
                groups.append(label or gid)
        result["layers_groups"] = groups[:50]

        # Conteggio elementi
        svg_counts = {}
        for tag in ["path", "rect", "circle", "line", "polyline", "polygon", "ellipse", "image"]:
            elements = list(root.iter(f"{{http://www.w3.org/2000/svg}}{tag}"))
            if elements:
                svg_counts[tag] = len(elements)
        result["element_counts"] = svg_counts

    except Exception as e:
        result["error"] = f"Errore parsing: {e}"

    return result


# ── PDF tecnico Parser ────────────────────────────────────────────────────────

def parse_pdf_technical(filepath) -> dict:
    """
    Parser per PDF tecnici.

    Estrae: testo pagina per pagina, tabelle (fitz), immagini → OCR.
    Specializzato per cartigli, distinte materiali, specifiche tecniche.
    """
    import fitz

    p = Path(filepath)
    result = {
        "filename": p.name,
        "format": "PDF_TECHNICAL",
        "filepath": str(filepath),
        "filesize_kb": round(p.stat().st_size / 1024, 1),
    }

    try:
        doc = fitz.open(str(filepath))
        result["pages"] = len(doc)

        pages_content = []
        tables = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if text:
                pages_content.append({
                    "page": page_num,
                    "text": text[:3000],  # cap per pagina
                })

            # Tabelle via fitz (PyMuPDF >= 1.23)
            try:
                for tab in page.find_tables().tables:
                    df = tab.to_pandas()
                    md = df.to_markdown(index=False)
                    if md and len(md) > 20:
                        tables.append({
                            "page": page_num,
                            "markdown": md[:2000],
                        })
            except Exception:
                pass

        result["pages_content"] = pages_content
        if tables:
            result["tables"] = tables

        doc.close()

    except Exception as e:
        result["error"] = f"Errore lettura: {e}"

    return result


# ── Router parser ─────────────────────────────────────────────────────────────

def parse_file(filepath) -> dict:
    """Seleziona il parser corretto in base all'estensione."""
    ext = Path(filepath).suffix.lower()

    if ext == ".dxf":
        return parse_dxf(filepath)
    elif ext in (".stp", ".step"):
        return parse_stp(filepath)
    elif ext == ".ifc":
        return parse_ifc(filepath)
    elif ext == ".stl":
        return parse_stl(filepath)
    elif ext == ".svg":
        return parse_svg(filepath)
    elif ext == ".pdf":
        return parse_pdf_technical(filepath)
    else:
        return {
            "filename": Path(filepath).name,
            "format": ext.upper().lstrip("."),
            "error": f"Formato {ext} non supportato dal drawings agent",
        }


# ==============================================================================
#  FASE 3A — SCHEDA STRUTTURATA → MARKDOWN
# ==============================================================================

def structured_to_markdown(data: dict) -> str:
    """
    Converte la scheda JSON strutturata in Markdown leggibile e indicizzabile.
    Ogni formato produce Markdown strutturato diverso ottimizzato per RAG.
    """
    lines = []
    fmt = data.get("format", "UNKNOWN")

    lines.append(f"# {data.get('filename', 'Sconosciuto')}")
    lines.append(f"*Formato: {fmt} | Dimensione: {data.get('filesize_kb', '?')} KB*")

    if "error" in data:
        lines.append(f"\n**Errore:** {data['error']}")
        return "\n".join(lines)

    # ── DXF
    if fmt == "DXF":
        lines.append(f"\nVersione DXF: {data.get('dxf_version', 'N/A')}")
        lines.append(f"Unità: {data.get('units', 'N/A')}")

        if "bounding_box" in data:
            bb = data["bounding_box"]
            lines.append(f"\nBounding box: {bb.get('size', 'N/A')}")

        if "layers" in data:
            lines.append(f"\n## Layer ({len(data['layers'])})")
            for name, info in list(data["layers"].items())[:40]:
                status = "ON" if info.get("is_on", True) else "OFF"
                count = data.get("entities_per_layer", {}).get(name, "?")
                lines.append(f"  - {name} (colore {info.get('color', '?')}, "
                             f"{status}, {count} entità)")

        if "entity_counts" in data:
            lines.append("\n## Entità geometriche")
            for etype, count in sorted(data["entity_counts"].items(), key=lambda x: -x[1]):
                lines.append(f"  - {etype}: {count}")

        if "dimensions" in data and data["dimensions"]:
            lines.append(f"\n## Quote dimensionali ({len(data['dimensions'])})")
            for dim in data["dimensions"][:60]:
                text = dim.get("text", "")
                meas = dim.get("measurement")
                if meas is not None:
                    lines.append(f"  - {text or '(auto)'} = {meas}")
                elif text:
                    lines.append(f"  - {text}")

        if "texts" in data and data["texts"]:
            lines.append(f"\n## Testi e annotazioni ({len(data['texts'])})")
            for t in data["texts"][:120]:
                lines.append(f"  {t}")

        if "blocks" in data and data["blocks"]:
            lines.append(f"\n## Blocchi/Componenti ({len(data['blocks'])})")
            for bname, count in sorted(data["blocks"].items(), key=lambda x: -x[1]):
                attribs = data.get("block_attributes", {}).get(bname, {})
                attr_str = ""
                if attribs:
                    attr_str = " | " + ", ".join(f"{k}={v}" for k, v in list(attribs.items())[:5])
                lines.append(f"  - {bname} ×{count}{attr_str}")

    # ── STEP/STP
    elif fmt == "STEP":
        if "description" in data:
            lines.append(f"\nDescrizione: {data['description']}")
        if "schema" in data:
            lines.append(f"Schema: {data['schema']}")
        lines.append(f"Unità: {data.get('units', 'N/A')}")

        if "products" in data:
            lines.append(f"\n## Prodotti/Componenti ({len(data['products'])})")
            for prod in data["products"]:
                lines.append(f"  - {prod}")

        if "entity_counts" in data:
            lines.append("\n## Entità geometriche")
            for name, count in sorted(data["entity_counts"].items(), key=lambda x: -x[1]):
                lines.append(f"  - {name}: {count}")

        if "materials" in data:
            lines.append("\n## Materiali")
            for mat in data["materials"]:
                lines.append(f"  - {mat}")

        if "geometric_analysis" in data:
            ga = data["geometric_analysis"]
            lines.append("\n## Analisi geometrica")
            for k, v in ga.items():
                label = k.replace("_", " ").capitalize()
                lines.append(f"  - {label}: {v}")

    # ── IFC
    elif fmt == "IFC":
        lines.append(f"\nSchema IFC: {data.get('schema', 'N/A')}")
        if "project_name" in data:
            lines.append(f"Progetto: {data['project_name']}")

        if "floors" in data:
            lines.append(f"\n## Piani ({len(data['floors'])})")
            for fl in data["floors"]:
                elev = f" — quota: {fl['elevation_m']}m" if "elevation_m" in fl else ""
                lines.append(f"  - {fl['name']}{elev}")

        if "elements" in data:
            lines.append("\n## Elementi edilizi")
            for name, info in data["elements"].items():
                lines.append(f"\n### {name} ({info['count']})")
                for n in info["names"][:10]:
                    lines.append(f"  - {n}")

        if "materials" in data:
            lines.append(f"\n## Materiali ({len(data['materials'])})")
            for mat in data["materials"]:
                lines.append(f"  - {mat}")

        if "sample_properties" in data:
            lines.append("\n## Proprietà (campione)")
            for pset_name, props in data["sample_properties"].items():
                lines.append(f"\n### {pset_name}")
                for k, v in props.items():
                    lines.append(f"  - {k}: {v}")

    # ── STL
    elif fmt == "STL":
        lines.append(f"\nTriangoli: {data.get('triangles', 'N/A')}")
        lines.append(f"Vertici: {data.get('vertices', 'N/A')}")
        lines.append(f"Watertight: {'Sì' if data.get('is_watertight') else 'No'}")

        if "bounding_box" in data:
            lines.append(f"Bounding box: {data['bounding_box'].get('extents', 'N/A')}")
        if "volume" in data:
            lines.append(f"Volume: {data['volume']}")
        if "surface_area" in data:
            lines.append(f"Area superficiale: {data['surface_area']}")
        if "separate_bodies" in data:
            lines.append(f"Corpi separati: {data['separate_bodies']}")
        if "convexity_ratio" in data:
            lines.append(f"Rapporto convessità: {data['convexity_ratio']}")

    # ── SVG
    elif fmt == "SVG":
        lines.append(f"\nDimensioni: {data.get('width', '?')} × {data.get('height', '?')}")
        lines.append(f"ViewBox: {data.get('viewBox', 'N/A')}")

        if "element_counts" in data:
            lines.append("\n## Elementi SVG")
            for tag, count in data["element_counts"].items():
                lines.append(f"  - <{tag}>: {count}")

        if "texts" in data and data["texts"]:
            lines.append(f"\n## Testi ({len(data['texts'])})")
            for t in data["texts"][:60]:
                lines.append(f"  {t}")

        if "layers_groups" in data and data["layers_groups"]:
            lines.append(f"\n## Layer/Gruppi ({len(data['layers_groups'])})")
            for g in data["layers_groups"]:
                lines.append(f"  - {g}")

    # ── PDF Tecnico
    elif fmt == "PDF_TECHNICAL":
        lines.append(f"\nPagine: {data.get('pages', 'N/A')}")

        if "pages_content" in data:
            for pc in data["pages_content"][:20]:
                lines.append(f"\n## Pagina {pc['page']}")
                lines.append(pc["text"][:2000])

        if "tables" in data:
            lines.append("\n## Tabelle estratte")
            for tab in data["tables"][:10]:
                lines.append(f"\n**Tabella pag. {tab['page']}:**")
                lines.append(tab["markdown"])

    return "\n".join(lines)


# ==============================================================================
#  FASE 3B — RENDER PREVIEW (DXF → PNG)
# ==============================================================================

def render_dxf_preview(filepath, output_path=None) -> str | None:
    """
    Renderizza un file DXF in PNG usando ezdxf + matplotlib.

    Utile per:
      - Dare contesto visivo (futuro: VLM per analisi visiva)
      - Mostrare preview nell'interfaccia
      - Validare che il DXF si apra correttamente

    Ritorna il path del PNG generato, o None se fallisce.
    """
    try:
        import ezdxf
        from ezdxf import recover
        from ezdxf.addons.drawing import matplotlib as ezdxf_mpl
    except ImportError:
        return None

    PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)

    p = Path(filepath)
    if output_path is None:
        h = hashlib.md5(str(filepath).encode()).hexdigest()[:8]
        output_path = PREVIEW_CACHE / f"{p.stem}_{h}.png"

    output_path = Path(output_path)

    # Cache hit: preview già generata e aggiornata
    if output_path.exists():
        try:
            if output_path.stat().st_mtime >= p.stat().st_mtime:
                return str(output_path)
        except OSError:
            pass

    try:
        doc, auditor = recover.readfile(str(filepath))
        if auditor.has_errors:
            print(f"  [preview] DXF con errori, provo comunque: {p.name}", flush=True)

        ezdxf_mpl.qsave(
            doc.modelspace(),
            str(output_path),
            dpi=150,
            bg="#FFFFFF",
        )

        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"  [preview] {output_path.name} ({size_kb:.1f} KB)", flush=True)
            return str(output_path)

    except Exception as e:
        print(f"  [preview] Errore render {p.name}: {e}", flush=True)

    return None


# ==============================================================================
#  FASE 4 — MARKDOWN CACHE + CHUNKING + INDEXING
# ==============================================================================

def _cache_path(filepath) -> Path:
    """Path deterministico nella cache per un file sorgente."""
    h = hashlib.md5(str(filepath).encode()).hexdigest()[:8]
    stem = re.sub(r"[^a-zA-Z0-9]", "_", Path(filepath).stem[:40]).strip("_")
    return MARKDOWN_CACHE / f"drw_{stem}_{h}.md"


def _cache_is_valid(filepath, cache_file: Path) -> bool:
    if not cache_file.exists():
        return False
    try:
        return cache_file.stat().st_mtime >= Path(filepath).stat().st_mtime
    except Exception:
        return False


def get_or_create_markdown(filepath) -> tuple[str, dict]:
    """
    Restituisce (markdown, structured_data) per un file tecnico.
    Usa cache Markdown su disco; rigenera se il sorgente cambia.
    """
    MARKDOWN_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(filepath)

    # Cache hit
    if _cache_is_valid(filepath, cache_file):
        try:
            md = cache_file.read_text(encoding="utf-8")
            return md, {}  # structured_data non serve se da cache
        except Exception:
            pass

    # Parse + convert
    structured = parse_file(filepath)
    markdown = structured_to_markdown(structured)

    if markdown and len(markdown.strip()) > 50:
        try:
            cache_file.write_text(markdown, encoding="utf-8")
            print(f"  [cache] {cache_file.name} ({len(markdown):,} char)", flush=True)
        except Exception as e:
            print(f"  [cache warning] {e}", flush=True)

    return markdown, structured


def chunk_text(text: str) -> list:
    """Spezza il testo in chunk con overlap."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks or [""]


def index_file(filepath) -> int:
    """
    Pipeline completa di indicizzazione per un file tecnico:
      1. Parse → scheda strutturata
      2. Markdown → cache
      3. Chunk → ChromaDB
      4. Preview PNG (solo DXF, non bloccante)
      5. Aggiorna memoria

    Funzione sincrona — il watcher è sync.
    """
    p = Path(filepath)
    ext = p.suffix.lower()

    if ext not in EXTENSIONS["drawings"]:
        return 0

    print(f"  [drawings] Elaborazione {p.name}...", flush=True)

    markdown, structured = get_or_create_markdown(filepath)

    if not markdown or not markdown.strip():
        return 0

    # Salva in memoria
    _update_drawing_memory(filepath, markdown, structured)

    # Rimuovi chunk stale
    try:
        existing = collection.get(where={"path": str(filepath)})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Metadata ricchi per ogni chunk
    base_meta = {
        "filename": p.name,
        "path":     str(filepath),
        "agent":    "drawings",
        "type":     "chunk",
        "ext":      ext,
        "format":   structured.get("format", ext.upper().lstrip(".")),
    }

    # Aggiungi info strutturate ai metadata (per filtri futuri)
    if "units" in structured:
        base_meta["units"] = structured["units"]
    if "products" in structured and structured["products"]:
        base_meta["products"] = ", ".join(structured["products"][:5])
    if "materials" in structured and isinstance(structured.get("materials"), list):
        base_meta["materials"] = ", ".join(structured["materials"][:5])

    chunks = chunk_text(markdown)
    for i, chunk in enumerate(chunks):
        meta = {**base_meta, "chunk": i}
        collection.upsert(
            documents=[chunk],
            ids=[f"{filepath}__c{i}"],
            metadatas=[meta],
        )

    # Render preview DXF (non bloccante, best-effort)
    if ext == ".dxf":
        try:
            render_dxf_preview(filepath)
        except Exception as e:
            print(f"  [preview] Skip: {e}", flush=True)

    return len(chunks)


def index_folder(folder):
    """Indicizza tutti i file tecnici in una cartella (ricorsivo)."""
    files = [f for f in Path(folder).rglob("*")
             if f.is_file() and f.suffix.lower() in EXTENSIONS["drawings"]]

    print(f"[Drawings] Trovati {len(files)} file...")
    total = 0
    for f in files:
        n = index_file(str(f))
        if n:
            print(f"  [OK] {f.name} → {n} chunk")
        else:
            print(f"  [--] {f.name} → saltato")
        total += n
    print(f"[Drawings] Completato — {total} chunk totali")


# ==============================================================================
#  MEMORIA DISEGNI
# ==============================================================================

def _load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"drawings": {}, "annotations": {}}


def _save_memory(memory: dict):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _update_drawing_memory(filepath, markdown: str, structured: dict):
    """Salva metadata per-disegno nella memoria persistente."""
    memory = _load_memory()
    p = Path(filepath)

    entry = {
        "filepath":   str(filepath),
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
        "words":      len(markdown.split()),
        "chars":      len(markdown),
        "size_kb":    round(p.stat().st_size / 1024, 1),
        "ext":        p.suffix.lower(),
        "format":     structured.get("format", ""),
    }

    # Aggiungi info chiave dalla scheda strutturata
    if "units" in structured:
        entry["units"] = structured["units"]
    if "products" in structured:
        entry["products"] = structured.get("products", [])[:10]
    if "materials" in structured and isinstance(structured.get("materials"), list):
        entry["materials"] = structured["materials"][:10]
    if "bounding_box" in structured:
        entry["bounding_box"] = structured["bounding_box"]
    if "entity_counts" in structured:
        entry["entity_stats"] = {
            k: v for k, v in sorted(
                structured["entity_counts"].items(), key=lambda x: -x[1]
            )[:10]
        }

    memory["drawings"][p.name] = entry
    _save_memory(memory)


# ==============================================================================
#  FILENAME DETECTION (come documents_agent)
# ==============================================================================

def detect_filename_filter(query: str) -> dict | None:
    """
    Analizza la query per trovare riferimenti a file specifici.
    Ritorna filtro ChromaDB 'where' per filename, o None.
    """
    try:
        all_meta = collection.get(include=["metadatas"])
        all_filenames = list({
            m["filename"] for m in all_meta["metadatas"]
            if m.get("filename") and m.get("type") != "semantic_card"
        })
    except Exception:
        return None

    if not all_filenames:
        return None

    def normalize(s: str) -> str:
        s = Path(s).stem if "." in s else s
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
        s = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', s)
        s = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', s)
        return re.sub(r'[_\-\s]+', ' ', s).strip().lower()

    query_norm = normalize(query)
    best_match = None
    best_score = 0

    for fname in all_filenames:
        fname_norm = normalize(fname)

        if fname_norm in query_norm or query_norm in fname_norm:
            score = len(fname_norm)
            if score > best_score:
                best_score = score
                best_match = fname
            continue

        fname_words = fname_norm.split()
        query_words = query_norm.split()
        if len(fname_words) >= 1 and all(fw in query_words for fw in fname_words):
            score = len(fname_norm)
            if score > best_score:
                best_score = score
                best_match = fname

    if best_match:
        print(f"  [smart-filter] '{query}' → '{best_match}'", flush=True)
        return {"filename": best_match}

    return None


# ==============================================================================
#  FASE 5 — SEARCH ASYNC (con semantic cards lazy)
# ==============================================================================

async def search(query: str) -> str:
    """Ricerca async con filename filtering + schede semantiche lazy."""
    where_filter = detect_filename_filter(query)
    return await analyzer.search_with_cards(
        collection, query, "drawings", n_results=6, where_filter=where_filter
    )


# ==============================================================================
#  FASE 6 — CROSS-REFERENCING
# ==============================================================================

async def _cross_reference(query: str, drawing_filename: str) -> str:
    """
    Cerca correlazioni con il disegno nelle altre collection:
      - documents: PDF catalogo, PPTX commerciali, DOCX specifiche
      - financial: dati di mercato eventualmente correlati

    Ritorna contesto aggiuntivo (stringa) o stringa vuota.
    """
    cross_context_parts = []

    # Cerca nella collection documents
    try:
        docs_client = chromadb.PersistentClient(path=CHROMA_PATHS["documents"])
        docs_collection = open_collection(docs_client, "documents", verify=False)

        # Cerca per nome prodotto / nome file nel contesto documenti
        search_terms = [
            drawing_filename,
            Path(drawing_filename).stem,
        ]

        # Estrai nomi prodotto dalla memoria
        loop = asyncio.get_running_loop()
        memory = await loop.run_in_executor(None, _load_memory)
        drawing_info = memory.get("drawings", {}).get(drawing_filename, {})
        products = drawing_info.get("products", [])
        if products:
            search_terms.extend(products[:3])

        combined_query = " ".join(search_terms[:4])

        results = docs_collection.query(
            query_texts=[combined_query],
            n_results=3,
            include=["documents", "metadatas"],
        )

        if results and results["documents"] and results["documents"][0]:
            docs_text = []
            for i, doc_text in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                source = meta.get("filename", "documento")
                docs_text.append(f"[Da {source}]: {doc_text[:500]}")

            if docs_text:
                cross_context_parts.append(
                    "=== DOCUMENTI CORRELATI ===\n" + "\n\n".join(docs_text)
                )

    except Exception as e:
        print(f"  [cross-ref] documents: {e}", flush=True)

    # Cerca nella collection financial (se esiste)
    try:
        if "financial" in CHROMA_PATHS:
            fin_client = chromadb.PersistentClient(path=CHROMA_PATHS["financial"])
            fin_collection = open_collection(fin_client, "financial", verify=False)

            results = fin_collection.query(
                query_texts=[Path(drawing_filename).stem],
                n_results=2,
                include=["documents", "metadatas"],
            )

            if results and results["documents"] and results["documents"][0]:
                fin_text = []
                for i, doc_text in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    source = meta.get("filename", "financial")
                    fin_text.append(f"[Da {source}]: {doc_text[:400]}")

                if fin_text:
                    cross_context_parts.append(
                        "=== DATI FINANZIARI CORRELATI ===\n" + "\n\n".join(fin_text)
                    )

    except Exception as e:
        print(f"  [cross-ref] financial: {e}", flush=True)

    return "\n\n".join(cross_context_parts)


# ==============================================================================
#  FASE 7 — SYSTEM PROMPT + ANSWER
# ==============================================================================

SYSTEM_PROMPT = """Sei un esperto analista di disegni tecnici e file CAD con accesso completo
ai disegni aziendali e ai documenti correlati.

REGOLE ASSOLUTE:
- Rispondi SEMPRE e SOLO in italiano, qualunque sia la lingua del file
- Basa le risposte ESCLUSIVAMENTE sulle informazioni estratte dai file nel contesto
- Cita sempre il file sorgente e l'elemento specifico (layer, entità, pagina)
- Non inventare mai dimensioni, quote o specifiche non presenti nei file

NUMERI, MISURE E SPECIFICHE — REGOLA CRITICA:
- Riporta i valori ESATTAMENTE come appaiono: nessuna trasformazione
- Preserva unità di misura (mm, cm, m, µm, kg, g, %, °C, bar, N, pz, €, ecc.)
- Non arrotondare mai, non convertire unità, non cambiare formato
- Tolleranze (±0.5, ±5%, max 3 mm) vanno sempre riportate col valore principale
- Range (es. 10–15 mm) vanno riportati interi

ANALISI FILE TECNICI:
- DXF: descrivi layer, entità geometriche, quote dimensionali, blocchi con attributi
- STP/STEP: identifica componenti, tipo di parte (rotazionale/prismatica), materiali
- IFC: descrivi struttura edificio, piani, elementi edilizi, materiali, proprietà BIM
- STL: riporta mesh stats (triangoli, volume, bounding box, watertight)
- SVG: descrivi elementi, testi, layer
- PDF tecnici: estrai dati da cartigli, distinte materiali, specifiche

CROSS-REFERENCING:
- Se il contesto include documenti correlati (PDF catalogo, PPTX, ecc.),
  usa queste informazioni per arricchire l'analisi del disegno
- Segnala corrispondenze tra disegno tecnico e documentazione commerciale

STRUTTURA RISPOSTE:
- Specifiche tecniche → tabella o lista ordinata con unità
- Confronti → colonne affiancate con fonte per ogni valore
- Riassunti → struttura gerarchica (sezioni → punti chiave)
- Ricerca valori → cita il contesto esatto del file"""


async def _build_user_prompt(question: str, context: str) -> str:
    """
    Costruisce il prompt utente includendo:
      - Indice disegni indicizzati
      - Contesto di ricerca (chunk + schede semantiche)
      - Cross-reference con altri agenti
    """
    loop = asyncio.get_running_loop()
    memory = await loop.run_in_executor(None, _load_memory)

    drawings = memory.get("drawings", {})
    mem_txt = ""

    if drawings:
        lines = []
        for name, info in list(drawings.items())[:25]:
            fmt = info.get("format", info.get("ext", ""))
            units = info.get("units", "")
            prods = ", ".join(info.get("products", [])[:3])
            extra = ""
            if units:
                extra += f" | {units}"
            if prods:
                extra += f" | {prods}"
            lines.append(f"  - {name} ({fmt}, {info.get('size_kb', '?')} KB{extra})")
        mem_txt += "\nDISEGNI INDICIZZATI:\n" + "\n".join(lines) + "\n"

    # Cross-referencing: cerca documenti correlati
    # Identifica il disegno principale dalla query
    drawing_filename = ""
    where_filter = detect_filename_filter(question)
    if where_filter and "filename" in where_filter:
        drawing_filename = where_filter["filename"]

    cross_context = ""
    if drawing_filename:
        try:
            cross_context = await _cross_reference(question, drawing_filename)
        except Exception as e:
            print(f"  [cross-ref] Errore: {e}", flush=True)

    prompt = f"""{mem_txt}
=== CONTENUTO DISEGNI TECNICI — FONTE PRIMARIA ===
{context}
"""

    if cross_context:
        prompt += f"\n{cross_context}\n"

    prompt += f"""
DOMANDA: {question}

RISPOSTA TECNICA (in italiano — dati solo dalle fonti fornite,
numeri e misure ESATTAMENTE come nel file sorgente):"""

    return prompt


# ── Answer (async, non-streaming) ─────────────────────────────────────────────

async def answer(question: str, context: str) -> str:
    user_prompt = await _build_user_prompt(question, context)
    return await chat_complete(
        model=ANSWER_MODEL,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.2,
    )


# ── Answer streaming ──────────────────────────────────────────────────────────

async def answer_stream(question: str, context: str):
    """Genera la risposta in streaming token-by-token."""
    user_prompt = await _build_user_prompt(question, context)
    async for chunk in chat_complete_stream(
        model=ANSWER_MODEL,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=0.2,
    ):
        yield chunk


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Indexing drawings in: {folder}")
    index_folder(folder)