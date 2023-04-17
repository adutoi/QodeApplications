#    (C) Copyright 2018 Anthony D. Dutoi
# 
#    This file is part of QodeApplications.
# 
#    QodeApplications is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    QodeApplications is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with QodeApplications.  If not, see <http://www.gnu.org/licenses/>.
#

def lines_all(raw):
	return [line for line in raw.split("\n")]

def lines(raw):
	return lines_all(raw)[:-1]	# assumes the last line is blank

def fields(raw):
	return [line.split() for line in raw]

def grid_all(raw):
	return fields(lines_all(raw))

def grid(raw):
	return fields(lines(raw))

def process(parse, lined):
	return [parse(n,line) for n,line in enumerate(lined)]

def extract(parse, raw):
	if isinstance(parse,tuple):  parse, preprocess = parse
	else:                        parse, preprocess = parse, grid
	if preprocess is None:  preprocessed = raw
	else:                   preprocessed = preprocess(raw)
	return process(parse, preprocessed)

def cast(types, begin=0):	# types is a dict:  {1:int, 2:float} where field 0 is the line enumerator
	def parse(n, fields):
		fields = [n+begin] + fields
		return tuple([ types[i](fields[i]) for i in types ])
	return parse

X = extract

