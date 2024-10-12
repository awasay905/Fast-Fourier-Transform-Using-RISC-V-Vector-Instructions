GCC_PREFIX = riscv32-unknown-elf
ABI = -march=rv32gcv -mabi=ilp32f
LINK = ./VeerFiles/link.ld

all: compile execute


clean: 
	rm -f ./VeerFiles/log.txt ./VeerFiles/program.hex ./VeerFiles/TEST.dis ./VeerFiles/TEST.exe
	
compile:
	$(GCC_PREFIX)-gcc $(ABI) -lgcc -T$(LINK) -o ./VeerFiles/TEST.exe vectorizedFFT.s -nostartfiles -lm
	$(GCC_PREFIX)-objcopy -O verilog ./VeerFiles/TEST.exe ./VeerFiles/program.hex
	$(GCC_PREFIX)-objdump -S ./VeerFiles/TEST.exe > ./VeerFiles/TEST.dis
	
	
execute:
	whisper -x ./VeerFiles/program.hex -s 0x80000000 --tohost 0xd0580000 -f ./VeerFiles/log.txt --configfile ./VeerFiles/whisper.json

allNV: compileNV executeNV


cleanNV: 
	rm -f ./VeerFiles/logNV.txt ./VeerFiles/programNV.hex ./VeerFiles/TESTNV.dis ./VeerFiles/TESTNV.exe
	
compileNV:
	$(GCC_PREFIX)-gcc $(ABI) -lgcc -T$(LINK) -o ./VeerFiles/TESTNV.exe FFT.s -nostartfiles -lm
	$(GCC_PREFIX)-objcopy -O verilog ./VeerFiles/TESTNV.exe ./VeerFiles/programNV.hex
	$(GCC_PREFIX)-objdump -S ./VeerFiles/TESTNV.exe > ./VeerFiles/TESTNV.dis
	
	
executeNV:
	whisper -x ./VeerFiles/programNV.hex -s 0x80000000 --tohost 0xd0580000 -f ./VeerFiles/log.txt --configfile ./VeerFiles/whisper.json



all2: compile2 execute2


clean2: 
	rm -f ./VeerFiles/log.txt ./VeerFiles/program.hex ./VeerFiles/TEST.dis ./VeerFiles/TEST.exe
	
compile2:
	$(GCC_PREFIX)-gcc $(ABI) -lgcc -T$(LINK) -o ./VeerFiles/TEST2.exe vectorizedFFT2.s -nostartfiles -lm
	$(GCC_PREFIX)-objcopy -O verilog ./VeerFiles/TEST2.exe ./VeerFiles/program2.hex
	$(GCC_PREFIX)-objdump -S ./VeerFiles/TEST2.exe > ./VeerFiles/TEST2.dis
	
	
execute2:
	whisper -x ./VeerFiles/program2.hex -s 0x80000000 --tohost 0xd0580000 -f ./VeerFiles/log.txt --configfile ./VeerFiles/whisper.json
