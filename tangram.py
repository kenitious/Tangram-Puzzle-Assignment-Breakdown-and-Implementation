import re
import math
from collections import defaultdict
import itertools
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

class TangramPuzzle:
    def __init__(self, filename):
        self.filename = filename
        self.pieces = []
        self.transformations = {}
        self.vertices = {}
        self.parse_file()
        self.process_transformations()
        self.calculate_vertices()

    def parse_coordinate(self, coord_str):
        """Parse coordinate strings into numerical values with exact handling"""
        coord_str = coord_str.strip().replace(' ', '')
        
        # Handle cases with leading + or multiple 0s
        coord_str = re.sub(r'^([+-]?)0+(\d)', r'\1\2', coord_str)
        
        # Patterns to match different coordinate formats
        patterns = [
            (r'^([+-]?\d*\.?\d+)$', lambda m: float(m.group(1))),  # Simple numbers
            (r'^([+-]?\d*\.?\d+)\*sqrt\(2\)$', lambda m: float(m.group(1)) * math.sqrt(2)),
            (r'^sqrt\(2\)$', lambda _: math.sqrt(2)),
            (r'^([+-]?\d*\.?\d+)\+(\d*\.?\d+)\*sqrt\(2\)$', 
             lambda m: float(m.group(1)) + float(m.group(2)) * math.sqrt(2)),
            (r'^([+-]?\d*\.?\d+)\-(\d*\.?\d+)\*sqrt\(2\)$',
             lambda m: float(m.group(1)) - float(m.group(2)) * math.sqrt(2)),
        ]

        for pattern, processor in patterns:
            match = re.fullmatch(pattern, coord_str)
            if match:
                return processor(match)
        
        # Handle cases like "a - sqrt(2)" or "a + b*sqrt(2)"
        if 'sqrt(2)' in coord_str:
            parts = coord_str.split('sqrt(2)')
            constant = 0.0
            sqrt_coeff = 0.0
            
            if parts[0]:
                pre = parts[0].replace('*', '').strip()
                if pre.endswith('+'):
                    sqrt_coeff = 1.0
                    pre = pre[:-1].strip()
                elif pre.endswith('-'):
                    sqrt_coeff = -1.0
                    pre = pre[:-1].strip()
                elif pre:
                    sqrt_coeff = float(pre)
                
                if pre and pre not in ['+', '-']:
                    constant = float(pre)
            
            if len(parts) > 1 and parts[1]:
                post = parts[1].strip()
                if post.startswith('+'):
                    constant += float(post[1:]) if post[1:] else 1.0
                elif post.startswith('-'):
                    constant -= float(post[1:]) if post[1:] else 1.0
                elif post:
                    constant += float(post)
            
            return constant + sqrt_coeff * math.sqrt(2)
        
        return 0.0

    def parse_file(self):
        """Parse the input .tex file with robust transformation handling"""
        with open(self.filename) as f:
            content = f.read()

        content = re.sub(r'%.*', '', content)  # Remove comments
        
        piece_pattern = r'\\PieceTangram\[TangSol\](?:<\s*([^>]*)\s*>)?\s*\(\s*{([^}]*)}\s*,\s*{([^}]*)}\s*\)\s*{([^}]*)}'
        pieces = re.finditer(piece_pattern, content)

        piece_counts = defaultdict(int)

        for match in pieces:
            options = match.group(1) or ""
            x_coord = match.group(2)
            y_coord = match.group(3)
            piece_type = match.group(4).strip()

            # Assign names according to specification
            if piece_type == 'TangGrandTri':
                piece_counts['TangGrandTri'] += 1
                name = f"Large triangle {piece_counts['TangGrandTri']}"
            elif piece_type == 'TangPetTri':
                piece_counts['TangPetTri'] += 1
                name = f"Small triangle {piece_counts['TangPetTri']}"
            else:
                name = {
                    'TangMoyTri': 'Medium triangle',
                    'TangCar': 'Square',
                    'TangPara': 'Parallelogram'
                }[piece_type]

            # Parse coordinates (handling cases like {-000.500})
            x = self.parse_coordinate(x_coord)
            y = self.parse_coordinate(y_coord)

            # Process transformations with proper ordering
            rotation = 0
            xflip = False
            yflip = False

            # Split options while handling spaces and commas
            option_list = [opt.strip().lower() for opt in re.split(r'\s*,\s*', options.strip()) if opt.strip()]
            
            for option in option_list:
                if 'rotate' in option:
                    # Handle rotation (including large values like 3780)
                    match_rotate = re.search(r'rotate\s*[:=]\s*(-?\d+)', option)
                    if match_rotate:
                        rotation = int(match_rotate.group(1))
                elif 'xscale' in option and '-1' in option:
                    xflip = True
                elif 'yscale' in option and '-1' in option:
                    yflip = True

            # Normalize transformations 
            if yflip:
                rotation += 180
                xflip = not xflip
            
            # Normalize rotation to 0-360 degrees
            rotation = rotation % 360
            
            self.pieces.append({
                'name': name,
                'type': piece_type,
                'x': x,
                'y': y,
                'rotation': rotation,
                'xflip': xflip
            })

    def process_transformations(self):
        """Store transformations in required format"""
        for piece in self.pieces:
            # Normalize rotation to 0-360
            rotation = piece['rotation'] % 360
            
            # Combine xflip and yflip into single xflip after rotation
            xflip = piece['xflip']
            if piece.get('yflip', False):
                rotation = (rotation + 180) % 360
                xflip = not xflip
                
            self.transformations[piece['name']] = {
                'rotate': rotation,
                'xflip': xflip
            }


    def calculate_vertices(self):
        """Calculate vertices for each piece after transformations with correct ordering"""
        sqrt2 = math.sqrt(2)
        
        for piece in self.pieces:
            piece_type = piece['type']
            tx, ty = piece['x'], piece['y']
            rotation = piece['rotation']
            xflip = piece['xflip']

            # Base vertices with exact dimensions
            if piece_type == 'TangGrandTri':  # Large triangle
                base = [(0, 0), (2, 0), (2, 2)]
            elif piece_type == 'TangMoyTri':   # Medium triangle
                base = [(0, 0), (2, 0), (1, 1)]
            elif piece_type == 'TangPetTri':   # Small triangle
                base = [(0, 0), (1, 0), (1, 1)]
            elif piece_type == 'TangCar':      # Square
                base = [(0, 0), (1, 0), (1, 1), (0, 1)]
            elif piece_type == 'TangPara':     # Parallelogram
                
                base = [
                    (0, 0),
                    (1, 0),
                    (1 + sqrt2/2, sqrt2/2),
                    (sqrt2/2, sqrt2/2)
                ]
            else:
                base = []

            transformed = []
            for x, y in base:
                # Apply rotation first (around origin)
                rad = math.radians(rotation)
                x_rot = x * math.cos(rad) - y * math.sin(rad)
                y_rot = x * math.sin(rad) + y * math.cos(rad)
                
                # Then apply x-flip (mirror over y-axis)
                if xflip:
                    x_rot = -x_rot
                
                # Finally apply translation
                transformed.append((x_rot + tx, y_rot + ty))

            # Find leftmost-topmost vertex (minimum x, maximum y)
            if not transformed:
                continue
                
            left_top = min(transformed, key=lambda v: (v[0], -v[1]))
            start_idx = transformed.index(left_top)
            
            # Reorder vertices clockwise starting from left_top
            # Calculate centroid for sorting
            cx = sum(v[0] for v in transformed) / len(transformed)
            cy = sum(v[1] for v in transformed) / len(transformed)
            
            # Sort by angle relative to centroid (clockwise)
            def angle_key(v):
                dx = v[0] - cx
                dy = v[1] - cy
                return math.atan2(dy, dx)
            
            ordered = sorted(transformed, key=angle_key, reverse=True)
            
            # Ensure left_top is first
            start_idx = ordered.index(left_top)
            ordered = ordered[start_idx:] + ordered[:start_idx]

            self.vertices[piece['name']] = {
                'vertices': ordered,
                'left_top': left_top
            }

    def format_coordinate(self, value):
        """Format coordinate value to match expected output"""
        sqrt2 = math.sqrt(2)
        tol = 1e-8
        
        # Check for simple integers
        if abs(value - round(value)) < tol:
            return str(int(round(value)))
        
        # Check for simple fractions
        for denom in [2]:
            num = round(value * denom)
            if abs(value - num/denom) < tol:
                return f"{num}/{denom}"
        
        # Check for √2 terms with fractional coefficients
        for denom in [2]:
            num = round(value * denom / sqrt2)
            if abs(value - num*sqrt2/denom) < tol:
                if num == 0:
                    return "0"
                return f"({num}/{denom})√2"
        
        # Check for a + b√2 form
        for a_denom in [2]:
            for a_num in range(-10, 10):
                a = a_num/a_denom
                b = (value - a)/sqrt2
                if abs(b - round(b*2)/2) < tol:
                    b_rounded = round(b*2)/2
                    a_str = self.format_coordinate(a)
                    b_str = self.format_coordinate(b_rounded*sqrt2)
                    if b_rounded > 0:
                        return f"{a_str} + {b_str}"
                    else:
                        return f"{a_str} - {b_str.replace('-','')}"
        
        return f"{value:.6f}"


    def __str__(self):
        """Produce output with exact coordinate formatting and correct ordering"""
        if not self.vertices:
            self.calculate_vertices()
        
        # Order pieces by left_top vertex (top to bottom, left to right)
        ordered_pieces = sorted(self.vertices.items(),
                             key=lambda item: (-item[1]['left_top'][1],  # y descending
                                              item[1]['left_top'][0]))  # x ascending
        
        output = []
        for name, data in ordered_pieces:
            # Remove numbers from piece names
            clean_name = ' '.join([word for word in name.split() if not word.isdigit()])
            
            vertices = data['vertices']
            vertex_strs = [f"({self.format_coordinate(x)}, {self.format_coordinate(y)})" 
                         for x, y in vertices]
            
            # Format with line wrapping 
            line = f"{clean_name} : [{', '.join(vertex_strs)}]"
            if len(line) > 60:
                parts = []
                current = vertex_strs[0]
                for v in vertex_strs[1:]:
                    if len(current) + len(v) + 2 <= 60:
                        current += ", " + v
                    else:
                        parts.append(current)
                        current = v
                parts.append(current)
                line = f"{clean_name} : [{parts[0]}"
                for p in parts[1:]:
                    line += ",\n" + " " * (len(clean_name) + 3) + p
                line += "]"
            
            output.append(line)
        
        return '\n'.join(output)

    def format_tex_coordinate(self, value):
        """Format coordinate for TikZ output with exact mathematical expressions"""
        sqrt2 = math.sqrt(2)
        tol = 1e-8
        
        # Check for simple integers
        if abs(value - round(value)) < tol:
            return str(int(round(value)))
        
        # Check for simple fractions
        for denom in [2, 3, 4]:
            num = round(value * denom)
            if abs(value - num/denom) < tol:
                if denom == 1:
                    return str(num)
                # Format fractions carefully
                if num < 0:
                    return f"({num}/{denom})"  # Negative fractions in parentheses
                return f"{num}/{denom}"
        
        # Check for √2 terms
        coeff = value / sqrt2
        if abs(coeff - round(coeff)) < tol:
            coeff = int(round(coeff))
            if coeff == 1:
                return "sqrt(2)"
            elif coeff == -1:
                return "-sqrt(2)"
            else:
                return f"{coeff}*sqrt(2)"
        
        # Check for a + b√2 form
        a = round(value)
        b = round((value - a)/sqrt2, 1)
        if abs(value - (a + b*sqrt2)) < tol:
            a_str = str(int(a)) if a == int(a) else str(a)
            if b == 0.5:
                b_str = "(1/2)*sqrt(2)"
            elif b == -0.5:
                b_str = "-(1/2)*sqrt(2)"
            else:
                b_int = int(b) if b == int(b) else b
                b_str = f"{b_int}*sqrt(2)"
            
            if a == 0:
                return b_str
            elif b > 0:
                return f"{a_str}+{b_str}" if a_str[0] != '-' else f"({a_str})+{b_str}"
            else:
                return f"{a_str}-{b_str.replace('-','')}" if a_str[0] != '-' else f"({a_str})-{b_str.replace('-','')}"
        
        # Fallback - use decimal format with sufficient precision
        return f"{value:.6f}"


    def draw_pieces(self, output_filename):
        """Create a .tex file displaying all pieces on a grid with exact coordinates"""
        if not self.vertices:
            self.calculate_vertices()

        # Find bounding box that includes all pieces
        all_vertices = [v for piece in self.vertices.values() for v in piece['vertices']]
        min_x = min(v[0] for v in all_vertices)
        max_x = max(v[0] for v in all_vertices)
        min_y = min(v[1] for v in all_vertices)
        max_y = max(v[1] for v in all_vertices)

        # Calculate grid bounds (extend by at least 1 square and less than 2 squares)
        def calculate_grid_bound(value, is_min):
            unit = 0.5  # 5mm grid
            grid_units = value / unit
            if is_min:
                return math.floor(grid_units - 1.999) * unit
            else:
                return math.ceil(grid_units + 1.999) * unit

        grid_min_x = calculate_grid_bound(min_x, True)
        grid_max_x = calculate_grid_bound(max_x, False)
        grid_min_y = calculate_grid_bound(min_y, True)
        grid_max_y = calculate_grid_bound(max_y, False)

        # Generate TikZ commands for each piece
        tikz_commands = []
        for name in sorted(self.vertices.keys(),
                        key=lambda n: (-self.vertices[n]['left_top'][1], 
                                    self.vertices[n]['left_top'][0])):
            vertices = self.vertices[name]['vertices']
            coord_pairs = []
            for x, y in vertices:
                x_str = self._format_exact_math(x)
                y_str = self._format_exact_math(y)
                coord_pairs.append(f"{{{x_str}}}")
                coord_pairs.append(f"{{{y_str}}}")

            # Create the draw command with proper coordinate grouping
            coords = []
            for i in range(0, len(coord_pairs), 2):
                coords.append(f"({coord_pairs[i]},{coord_pairs[i+1]})")
            draw_cmd = "\\draw[ultra thick] " + " -- ".join(coords) + " -- cycle;"
            tikz_commands.append(draw_cmd)

        # Create the complete TikZ picture with exact formatting
        tikz_code = f"""\\documentclass{{standalone}}
    \\usepackage{{tikz}}
    \\begin{{document}}

    \\begin{{tikzpicture}}
    \\draw[step=5mm] ({self._format_grid_coord(grid_min_x)},{self._format_grid_coord(grid_min_y)}) grid ({self._format_grid_coord(grid_max_x)},{self._format_grid_coord(grid_max_y)});
    {chr(10).join(tikz_commands)}
    \\fill[red] (0,0) circle (3pt);
    \\end{{tikzpicture}}

    \\end{{document}}"""

        # Write to output file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(tikz_code)

    def _format_grid_coord(self, value):
        """Format grid coordinates to simplest form"""
        if value == int(value):
            return str(int(value))
        # Handle .5 cases specially
        if abs(value * 2 - round(value * 2)) < 1e-8:
            if round(value * 2) % 2 == 0:
                return str(int(value))
            return f"{int(value)}.5" if value > 0 else f"-{abs(int(value))}.5"
        return f"{value:.1f}"

    def _format_exact_math(self, value):
        """Convert decimal values to exact mathematical expressions with sqrt(2)"""
        sqrt2 = math.sqrt(2)
        tol = 1e-6
        
        # Check if value is exactly 0
        if abs(value) < tol:
            return "0"
        
        # Check for simple integers
        if abs(value - round(value)) < tol:
            return str(int(round(value)))
        
        # Check for simple fractions (1/2, 3/2, etc.)
        for denom in [2, 3, 4]:
            num = round(value * denom)
            if abs(value - num/denom) < tol:
                if denom == 1:
                    return str(num)
                return f"{num}/{denom}"
        
        # Check for √2 terms with fractional coefficients
        for denom in [1, 2, 4]:
            num = round(value * denom / sqrt2)
            if abs(value - num*sqrt2/denom) < tol:
                if num == 0:
                    return "0"
                if denom == 1:
                    return f"{num}*sqrt(2)"
                elif denom == 2:
                    return f"{num}/2*sqrt(2)"
                else:
                    return f"{num}/4*sqrt(2)"
        
        # Check for a + b√2 form (more precise calculation)
        # Try all integer and half-integer values for a
        for a in [x/2 for x in range(-20, 20)]:
            b = (value - a)/sqrt2
            b_rounded = round(b*4)/4  # Round to nearest 0.25
            
            if abs(b - b_rounded) < tol:
                # Format a term
                if a == int(a):
                    a_str = str(int(a))
                else:
                    a_str = f"{int(a*2)}/2"
                
                # Format b term
                if b_rounded == 0:
                    return a_str
                elif b_rounded == 0.5:
                    b_str = "1/2*sqrt(2)"
                elif b_rounded == -0.5:
                    b_str = "-1/2*sqrt(2)"
                elif b_rounded == 0.25:
                    b_str = "1/4*sqrt(2)"
                elif b_rounded == -0.25:
                    b_str = "-1/4*sqrt(2)"
                elif b_rounded == int(b_rounded):
                    b_str = f"{int(b_rounded)}*sqrt(2)"
                else:
                    b_str = f"{int(b_rounded*2)}/2*sqrt(2)"
                
                # Combine terms
                if a == 0:
                    return b_str
                elif b_rounded > 0:
                    return f"{a_str}+{b_str}"
                else:
                    return f"{a_str}-{b_str.replace('-','')}"
        
        # Fallback - format as decimal if no exact form found
        return f"{value:.6f}"


    def format_tex_float(self, value):
        """Format float values for grid coordinates in TikZ"""
        if value == int(value):
            return str(int(value))
        return f"{value:.1f}"


    def draw_outline(self, output_filename):
        """Create a .tex file displaying only the outer outline of the puzzle"""
        if not self.vertices:
            self.calculate_vertices()

        # 1. Collect all edges from all pieces with precise comparison
        edge_counts = defaultdict(int)
        for piece_data in self.vertices.values():
            vertices = piece_data['vertices']
            for i in range(len(vertices)):
                p1 = vertices[i-1]
                p2 = vertices[i]
                # Create normalized edge representation (sorted points)
                edge = tuple(sorted([
                    (round(p1[0], 8), round(p1[1], 8)),
                    (round(p2[0], 8), round(p2[1], 8))
                ]))
                edge_counts[edge] += 1

        # 2. Identify outline edges (edges used by only one piece)
        outline_edges = [edge for edge, count in edge_counts.items() if count == 1]

        # 3. Reconstruct the complete outline path
        outline_paths = []
        while outline_edges:
            # Start new path with leftmost-topmost edge
            outline_edges.sort(key=lambda e: (min(v[0] for v in e), -max(v[1] for v in e)))
            current_edge = outline_edges.pop(0)
            current_path = list(current_edge)
            
            # Try to connect edges in both directions
            for direction in [1, -1]:
                path_complete = False
                while not path_complete and outline_edges:
                    last_point = current_path[-1] if direction == 1 else current_path[0]
                    connected = False
                    
                    # Find connecting edge
                    for i, edge in enumerate(outline_edges):
                        if last_point in edge:
                            # Determine next point
                            next_point = edge[1] if edge[0] == last_point else edge[0]
                            
                            # Add to path
                            if direction == 1:
                                current_path.append(next_point)
                            else:
                                current_path.insert(0, next_point)
                            
                            outline_edges.pop(i)
                            connected = True
                            break
                    
                    if not connected:
                        path_complete = True
                
                # Check if path forms a loop
                if len(current_path) > 2 and current_path[0] == current_path[-1]:
                    path_complete = True
            
            outline_paths.append(current_path)

        # 4. Clean paths (remove duplicates and ensure proper closure)
        clean_paths = []
        for path in outline_paths:
            clean_path = []
            for point in path:
                if not clean_path or point != clean_path[-1]:
                    clean_path.append(point)
            
            # Close the path if endpoints are close enough
            if len(clean_path) > 2 and math.dist(clean_path[0], clean_path[-1]) < 1e-6:
                clean_path[-1] = clean_path[0]  # Make exact match
                clean_paths.append(clean_path)
            elif len(clean_path) > 1:
                clean_paths.append(clean_path)

        # 5. Calculate grid bounds (same as draw_pieces)
        all_vertices = [v for path in clean_paths for v in path] or \
                    [v for piece in self.vertices.values() for v in piece['vertices']]
        
        min_x = min(v[0] for v in all_vertices)
        max_x = max(v[0] for v in all_vertices)
        min_y = min(v[1] for v in all_vertices)
        max_y = max(v[1] for v in all_vertices)

        def calculate_grid_bound(value, is_min):
            unit = 0.5  # 5mm grid
            grid_units = value / unit
            if is_min:
                return math.floor(grid_units - 1.999) * unit
            return math.ceil(grid_units + 1.999) * unit

        grid_min_x = calculate_grid_bound(min_x, True)
        grid_max_x = calculate_grid_bound(max_x, False)
        grid_min_y = calculate_grid_bound(min_y, True)
        grid_max_y = calculate_grid_bound(max_y, False)

        # 6. Generate TikZ code with exact formatting
        tikz_lines = [
            r"\documentclass{standalone}",
            r"\usepackage{tikz}",
            r"\begin{document}",
            "",
            r"\begin{tikzpicture}[x=1cm,y=1cm]",
            f"\\draw[step=5mm] ({self._format_grid_coord(grid_min_x)},{self._format_grid_coord(grid_min_y)}) "
            f"grid ({self._format_grid_coord(grid_max_x)},{self._format_grid_coord(grid_max_y)});"
        ]

        for path in clean_paths:
            if len(path) < 2:
                continue
                
            path_line = r"\draw[ultra thick]"
            for i, (x, y) in enumerate(path):
                x_str = self._format_exact_math(x)
                y_str = self._format_exact_math(y)
                coord = f"{{{x_str}}}, {{{y_str}}}"
                
                if i == 0:
                    path_line += f" ({coord})"
                else:
                    path_line += f" -- ({coord})"
            
            # Close the path if it forms a loop
            if len(path) > 2 and path[0] == path[-1]:
                path_line += " -- cycle;"
            else:
                path_line += ";"
            
            tikz_lines.append(path_line)

        # Add origin marker and close document
        tikz_lines.extend([
            r"\fill[red] (0,0) circle (3pt);",
            r"\end{tikzpicture}",
            "",
            r"\end{document}"
        ])

        # Write to file with consistent formatting
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(tikz_lines))

    def solve_tangram_puzzle(self, filename):
        """Main function to solve a tangram puzzle from an outline file"""
        outline_puzzle = TangramPuzzle(filename)
        target_polygon = self._vertices_to_polygon(outline_puzzle._get_all_vertices())
        
        standard_pieces = self._get_standard_pieces()
        all_transforms = self._generate_all_transforms(standard_pieces)
        
        solution = self._solve_with_backtracking(all_transforms, target_polygon)
        
        if solution:
            solved_puzzle = TangramPuzzle(filename)
            solved_puzzle.pieces = [{
                'name': p['name'],
                'type': p['type'],
                'x': p['position'][0],
                'y': p['position'][1],
                'rotation': p['rotation'],
                'xflip': p['flip']
            } for p in solution]
            solved_puzzle.calculate_vertices()
            return solved_puzzle
        
        return outline_puzzle

    def _get_standard_pieces(self):
        """Define the standard 7 tangram pieces"""
        return [
            {'name': 'Large triangle 1', 'type': 'TangGrandTri', 
             'base': [(0, 0), (2, 0), (2, 2)], 'area': 2.0},
            {'name': 'Large triangle 2', 'type': 'TangGrandTri',
             'base': [(0, 0), (2, 0), (2, 2)], 'area': 2.0},
            {'name': 'Medium triangle', 'type': 'TangMoyTri',
             'base': [(0, 0), (2, 0), (1, 1)], 'area': 1.0},
            {'name': 'Small triangle 1', 'type': 'TangPetTri',
             'base': [(0, 0), (1, 0), (1, 1)], 'area': 0.5},
            {'name': 'Small triangle 2', 'type': 'TangPetTri',
             'base': [(0, 0), (1, 0), (1, 1)], 'area': 0.5},
            {'name': 'Square', 'type': 'TangCar',
             'base': [(0, 0), (1, 0), (1, 1), (0, 1)], 'area': 1.0},
            {'name': 'Parallelogram', 'type': 'TangPara',
             'base': [(0, 0), (1, 0), (1 + math.sqrt(2)/2, math.sqrt(2)/2), 
                     (math.sqrt(2)/2, math.sqrt(2)/2)], 'area': 1.0}
        ]

    def _generate_all_transforms(self, standard_pieces):
        """Generate all possible transformations for each piece"""
        all_transforms = []
        for piece in standard_pieces:
            for flip in [False, True]:
                for rotation in [0, 45, 90, 135, 180, 225, 270, 315]:
                    all_transforms.append({
                        'name': piece['name'],
                        'type': piece['type'],
                        'base': piece['base'],
                        'flip': flip,
                        'rotation': rotation,
                        'area': piece['area']
                    })
        return all_transforms

    def _solve_with_backtracking(self, all_pieces, target_polygon, current_solution=None, used_indices=None):
        """Backtracking solver with optimizations"""
        if current_solution is None:
            current_solution = []
        if used_indices is None:
            used_indices = set()
        
        # Base case: all pieces placed
        if len(used_indices) == 7:
            if self._validate_solution(current_solution, target_polygon):
                return current_solution
            return None
        
        # Try each piece not yet used
        for i, piece in enumerate(all_pieces):
            if i not in used_indices and len(used_indices) < 7:
                # Generate possible positions for this piece
                for position in self._generate_possible_positions(piece, current_solution, target_polygon):
                    transformed_piece = {
                        **piece,
                        'position': position,
                        'vertices': self._transform_piece(piece['base'], piece['rotation'], piece['flip'], position)
                    }
                    
                    # Check area constraint
                    current_area = sum(p['area'] for p in current_solution) + piece['area']
                    if abs(current_area - 8.0) > 1e-6:  # Total tangram area is 8
                        continue
                    
                    # Check for overlaps
                    if self._has_overlap(transformed_piece, current_solution):
                        continue
                    
                    # Recursive call
                    result = self._solve_with_backtracking(
                        all_pieces,
                        target_polygon,
                        current_solution + [transformed_piece],
                        used_indices | {i}
                    )
                    if result:
                        return result
        return None

    def _generate_possible_positions(self, piece, current_solution, target_polygon):
        """Generate possible positions to place the piece"""
        # First placement - try center positions
        if not current_solution:
            for x in range(-2, 3):
                for y in range(-2, 3):
                    yield (x, y)
        else:
            # Subsequent placements - try near existing pieces
            for placed_piece in current_solution:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        yield (placed_piece['position'][0] + dx, placed_piece['position'][1] + dy)

    def _transform_piece(self, base, rotation, flip, position):
        """Transform piece vertices according to rotation, flip and position"""
        transformed = []
        for x, y in base:
            # Apply flip
            if flip:
                x = -x
            # Apply rotation
            rad = math.radians(rotation)
            x_rot = x * math.cos(rad) - y * math.sin(rad)
            y_rot = x * math.sin(rad) + y * math.cos(rad)
            # Apply translation
            transformed.append((x_rot + position[0], y_rot + position[1]))
        return transformed

    def _has_overlap(self, piece, current_solution):
        """Check if a piece overlaps with already placed pieces using polygon intersection"""
        piece_poly = Polygon(piece['vertices'])
        for placed_piece in current_solution:
            placed_poly = Polygon(placed_piece['vertices'])
            if piece_poly.intersects(placed_poly):
                return True
        return False

    def _validate_solution(self, solution, target_polygon):
        """Validate if the solution matches the target outline using polygon comparison"""
        solution_polygons = [Polygon(p['vertices']) for p in solution]
        combined_solution = unary_union(solution_polygons)
        return combined_solution.equals(target_polygon)

    def _vertices_to_polygon(self, vertices):
        """Convert a list of vertices to a Shapely polygon"""
        return Polygon(vertices)

    def _get_all_vertices(self):
        """Get all vertices from all pieces in the puzzle"""
        all_vertices = []
        for piece_data in self.vertices.values():
            all_vertices.extend(piece_data['vertices'])
        return all_vertices

# Standalone function that can be imported
def solve_tangram_puzzle(filename):
    """Solve a tangram puzzle from an outline file"""
    puzzle = TangramPuzzle(filename)
    return puzzle.solve_tangram_puzzle(filename)