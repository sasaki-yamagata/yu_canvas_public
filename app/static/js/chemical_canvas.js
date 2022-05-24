

// Copyright (c) 2019 Yamagata University
// This function will be called after the JavaScriptApplet code has been loaded.
function jsmeOnLoad() {
	jsmeApplet = new JSApplet.JSME("jsme", "400px", "400px", {"options" : "oldlook,star"});
	jsmeApplet.setAfterStructureModifiedCallback(updatePrediction);
}


// This function will be called every time after structure is modified.
function updatePrediction(event) {

	const mol = Kekule.IO.loadFormatData(event.src.molFile(false), 'mol');
	// showMolecule(mol)

	// 原子の数が０の時とそうでないとき
	if (mol.getNodeCount() == 0) {
		// update table
		document.getElementById("lumo").innerHTML = "-";
		document.getElementById("homo").innerHTML = "-";
	} else {

		// basic information
		// 分子情報を取得
		const formula = mol.calcFormula();	
		const weight = molecular_weight(formula)
		document.getElementById("formula").innerHTML = formula.getText();
		// toFixed(1)は四捨五入
		document.getElementById("weight").innerHTML = weight.toFixed(1);


		// standardize
		addImplicitHydrogens(mol);

    
		// const weak_atom = weakAtomList(mol, weak_bond)
		// 木構造の記述子作成
		const symbols = symbolArray(mol);
		const neighbors = neighborIndexArray(mol);
		const desc1 = nextLevelExpressionArray(symbols, symbols, neighbors, false);
		const featureValues1 = descriptorCounts(desc1);
		const desc2 = nextLevelExpressionArray(symbols, desc1, neighbors);
		const featureValues2 = descriptorCounts(desc2);

		// フラグメントの記述子作成
		const fsmiles = fragmentList(mol, max_ring=8)
		const featureValuesFrag = descriptorCounts(fsmiles)

		// 木構造とフラグメントの記述子を合わせる
		const feature = {...featureValues2, ...featureValuesFrag}
		feature['weight'] = weight
		console.log(feature)
		// pythonにデータを送信
		$.ajax({
				type: "POST",
				url: "/model",
				data: feature, // post a json data.
				async: false,
				dataType: "json",
				// 成功した時の処理
				success: function(response) {
						let pre_orb = response
						console.log(pre_orb)
								// predict HOMO and LUMO energies
						let level = 2;
								// 仮定義
						console.log(pre_orb)
						let homo = pre_orb['homo'] * -1
						let lumo = pre_orb['lumo'] * -1
						if (isNaN(lumo)) {
							// document.getElementById("level").innerHTML = "n/a";
							document.getElementById("lumo").innerHTML = "n/a";
							document.getElementById("homo").innerHTML = "n/a";
						} else {
							// document.getElementById("level").innerHTML = level;
							document.getElementById("lumo").innerHTML = lumo.toFixed(1) + " eV";
							document.getElementById("homo").innerHTML = homo.toFixed(1) + " eV";
						}
						// update energy diagram
						updateEnergyDiagram(lumo, homo);
						// Receive incremented number.
				}, 
				// 失敗した時の処理
				error: function(error) {
						console.log("Error occurred in keyPressed().");
						console.log(error);
				}
			})
	}
}



let atomic_weights = [0.0, 1.008, 4.002602, 6.94, 9.0121831, 10.81, 12.011, 14.007, 15.999, 18.99840316, 20.1797, 22.98976928, 24.305, 26.9815385, 28.085, 30.973762, 32.06, 35.45, 39.948, 39.0983, 40.078, 44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 95.95, 97.0, 101.07, 102.9055, 106.42, 107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.905452, 137.327, 138.90547, 140.116, 140.90766, 144.242, 145.0, 150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.045, 174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592, 204.38, 207.2, 208.9804, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0377, 231.03588, 238.02891, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0, 258.0, 259.0, 262.0, 267.0, 270.0, 269.0, 270.0, 270.0, 278.0, 281.0, 281.0, 285.0, 286.0, 289.0, 289.0, 293.0, 293.0, 294.0];

function molecular_weight(formula) {
	let w = 0.0;
	formula.sections.forEach(function(x) {
		w += atomic_weights[x["obj"].atomicNumber] * x["count"];
	});
	return w;
}

function updateEnergyDiagram(lumo, homo) {
	let diagram = document.getElementById("energy_diagram");
	if (diagram && diagram.getContext) {
		let c = diagram.getContext("2d");
		let x0 = 125; // origin
		let y0 = 30; // origin
		let s = 20; // scale
		let x, y;

		c.clearRect(0, 0, diagram.width, diagram.height);

		c.strokeStyle = "rgb(00, 00, 00)";
		c.fillStyle = "rgb(00, 00, 00)";
		c.font = "12pt Arial";
		c.textAlign = "right";

		// vacuum level
		horizontalDashedLine(c, x0, y0, 80);
		c.fillText("Vacuum (0 eV)", x0-10, y0+6);

		// standard hydrogen electrode (HSE)
		horizontalDashedLine(c, x0, y0+4.44*s, 80);
		c.fillText("SHE (4.44 eV)", x0-10, y0+4.44*s+6);

		c.strokeStyle = "rgb(255, 00, 00)";
		c.fillStyle = "rgb(255, 00, 00)";
		c.font = "12pt Arial";
		c.textAlign = "left";

		// LUMO
		horizontalLine(c, x0+20, y0+lumo*s, 80);
		c.fillText("LUMO ("+lumo.toFixed(1)+" eV)", x0+100+10, y0+lumo*s+6);

		// HOMO
		horizontalLine(c, x0+20, y0+homo*s, 80);
		c.fillText("HOMO ("+homo.toFixed(1)+" eV)", x0+100+10, y0+homo*s+6);

		// HOMO-LUMO gap
		verticalArrow(c, x0+90, y0+homo*s, (homo-lumo)*s);
		c.fillText("Gap ("+(homo-lumo).toFixed(1)+" eV)", x0+90+10, y0+(homo+lumo)*0.5*s+6);
	}
}

function verticalArrow(c, x, y, height) {
	c.beginPath();
	c.moveTo(x, y);
	c.lineTo(x, y-height);
	c.lineTo(x-6, y-height+8);
	c.moveTo(x, y-height);
	c.lineTo(x+6, y-height+8);
	c.closePath();
	c.stroke();
}

function horizontalLine(c, x, y, length) {
	c.beginPath();
	c.moveTo(x, y);
	c.lineTo(x+length, y);
	c.closePath();
	c.stroke();
}

function horizontalDashedLine(c, x, y, length) {
	c.beginPath();
	for (let i = 0; i < length; i = i+8) {
		c.moveTo(x+i, y);
		c.lineTo(x+Math.min(i+4, length), y);
	}
	c.closePath();
	c.stroke();
}




// 水素追加
function addImplicitHydrogens(mol) {
	const n_atoms = mol.getNodeCount();
	for (let i = 0; i < n_atoms; i++) {
		const node = mol.getNodeAt(i);
		const n_implicit_hydrogens = node.getImplicitHydrogenCount();
		for (let j = 0; j < n_implicit_hydrogens; j++) {
			const new_atom = new Kekule.Atom("", 1);
			const new_bond = new Kekule.Bond("", [node, new_atom], 1, 2, "COVALENT");
			mol.appendNode(new_atom);
			mol.appendConnector(new_bond);
		}
	}
}
// 切る予定の結合を集める
function fragmentList(mol, max_ring=8) {
	// 切る予定の結合集合
	let weak_bonds = new Set();

	// 単結合を weak_bonds に追加
	const connect_count = mol.getConnectorCount();

	for (let i = 0; i < connect_count; i++) {
		const connector = mol.getConnectorAt(i);
		if (connector.getBondOrder() == 1) {
			weak_bonds.add(connector);
		}	
	}
	const rings = mol.findAllRings();

	// max_ring 以下の環に含まれる結合は weak_bonds から除外
	for (let i = 0, l = rings.length; i < l; ++i)
	{
		const ring = rings[i];
		const connectorCount = ring.connectors.length;
		if (connectorCount <= max_ring) {
			for (let x = 0, y = connectorCount; x < y; ++x ) {
				let b = ring.connectors[x];

				if (weak_bonds.has(b)) {
					weak_bonds.delete(b);
				}
			}
		}
	}

	removed_bonds = new Set();

	for (let b of weak_bonds) {

		const a0 = b.getConnectedObjs()[0];
		const a1 = b.getConnectedObjs()[1];
		const n0 = a0.getAtomicNumber();
		const n1 = a1.getAtomicNumber();

		// 両方とも非炭素原子である場合は除外する。
		if (n0 != 6 && n1 != 6) {
			removed_bonds.add(b);
		// 末端水素は除外する。
		} else if (a0.linkedConnectors.length == 1 && a0.symbol == "H") {
			removed_bonds.add(b);
		} else if (a1.linkedConnectors.length == 1 && a1.symbol == "H" ) {
			removed_bonds.add(b);
		// 末端原子は除外する。
		}	else if (a0.linkedConnectors.length == 1 || a1.linkedConnectors.length == 1) {
			removed_bonds.add(b)
		// -CH2-CH2- は除外する。
		} else if (n0 == 6 && n1 == 6) {

			if (a0.linkedConnectors.length == 4 && a1.linkedConnectors.length == 4) {
				let h_count = 0;
				
				for (let neighbor of a0.linkedObjs) {
					if (neighbor.getAtomicNumber() == 1) {
						h_count += 1;				
					}
				}

				if (h_count == 2) {
					removed_bonds.add(b);			
					continue;
				}

				h_count = 0
				for (let neighbor of a1.linkedObjs) {
					if (neighbor.getAtomicNumber() == 1) {
						h_count += 1;				
					}
				}
				if (h_count == 2) {
					removed_bonds.add(b);
					continue;
				}
			}
		}
	}
	
	let weak_bonds_update = new Set([...weak_bonds].filter(b => (!removed_bonds.has(b))));
	
	// 切る
	for (let b of weak_bonds_update) {
		mol.removeConnector(b);
	}
	const smiles = Kekule.IO.saveFormatData(mol, 'smi');
	const fsmiles = smiles.split('.');
	return fsmiles



}


// 原子の配列を返す
function symbolArray(mol) {
	const n_atoms = mol.getNodeCount();
	const symbols = new Array(n_atoms);
	for (let i = 0; i < n_atoms; i++) {
		symbols[i] = mol.getNodeAt(i).symbol;
	}
	return symbols;
}
// 隣接する原子を配列を返す
function neighborIndexArray(mol) {
	const n_atoms = mol.getNodeCount();
	for (let i = 0; i < n_atoms; i++) {
		mol.getNodeAt(i).id = i;
	}
	const neighbors = new Array(n_atoms);
	for (let i = 0; i < n_atoms; i++) {
		neighbors[i] = new Array();
	}
	const n_conn = mol.getConnectorCount();
	for (let i = 0; i < n_conn; i++) {
		const conn = mol.getConnectorAt(i);
		
		const objs = conn.getConnectedObjs();
		const atom0 = parseInt(objs[0].id);
		const atom1 = parseInt(objs[1].id);

		neighbors[atom0].push(atom1);
		neighbors[atom1].push(atom0);
	}
	return neighbors;
	
}


function nextLevelExpressionArray(symbols, preLevelExp, neighbors, addParentheses = true) {
	const n = symbols.length;
	const nextLevelExp = new Array(n);
	for (let i = 0; i < n; i++) {
		const m = neighbors[i].length;
		const exp = new Array(m);
		for (let j = 0; j < m; j++) {
			exp[j] = preLevelExp[neighbors[i][j]];
		}
		exp.sort();
		nextLevelExp[i] = symbols[i] + "-";
		for (let j = 0; j < m; j++) {
			if (addParentheses) {
				nextLevelExp[i] += "(" + exp[j] + ")";
			} else {
				nextLevelExp[i] += exp[j];
			}
		}
	}
	return nextLevelExp;
}

function descriptorCounts(descriptors) {
	const counts = {};
	for (const d of descriptors) {
		if (d in counts) {
			counts[d]++;
		} else {
			counts[d] = 1;
		}
	}
	return counts;
}

// predict HOMO or LUMO from coefficients and feature values
function predict(coef, features) {
	let result = coef['intercept'];
	for (const key in features) {
		if (key in coef) {
			result += coef[key] * features[key];
		} else {
			return NaN;
		}     
	}
	return result;
}

