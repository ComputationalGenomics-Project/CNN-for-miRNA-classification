def u2t():
	path = raw_input("Please input the path of miRNA fasta file: ")
	file = open(path, 'r')
	fileout = open("out.txt", 'w')
	count = 0
	minlen = 30
	maxlen = 0
	while (True):
		line = file.readline()
		if (line == ''):
			break
		count += 1
		if (count % 2 == 1):
			continue
		minlen = min(minlen, len(line) - 1)
		maxlen = max(maxlen, len(line) - 1)
		newMir = convert(line)
		fileout.write(newMir)
	print "minimum length is: ", minlen
	print "maximum length is: ", maxlen
	file.close()
	fileout.close()


def convert(mir):
	if (mir == None or len(mir) == 0):
		mir
	else:
		for i in range(len(mir)):
			if (mir[i] == 'U' or mir[i] == 'u'):
				mir = mir[:i] + 'T' + mir[i+1:]
	return mir


def main():
	u2t()

if __name__ == "__main__":
	main() 