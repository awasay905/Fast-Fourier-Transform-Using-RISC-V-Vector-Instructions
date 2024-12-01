GCC_PREFIX = riscv32-unknown-elf
ABI = -march=rv32gcv_zbb_zbs -mabi=ilp32f
LINK = ./veer/link.ld
CODEFOLDER = ./src/assembly
TEMPPATH = ./veer/tempFiles

clean: cleanV cleanV2 cleanNV cleanNV2


allV: compileV executeV

cleanV: 
	rm -f $(TEMPPATH)/logV.txt  $(TEMPPATH)/programV.hex  $(TEMPPATH)/TESTV.dis  $(TEMPPATH)/TESTV.exe
	
compileV:
	$(GCC_PREFIX)-gcc $(ABI) -lgcc -T$(LINK) -o  $(TEMPPATH)/TESTV.exe $(CODEFOLDER)/FFT_V.s -nostartfiles -lm
	$(GCC_PREFIX)-objcopy -O verilog  $(TEMPPATH)/TESTV.exe  $(TEMPPATH)/programV.hex
	$(GCC_PREFIX)-objdump -S  $(TEMPPATH)/TESTV.exe >  $(TEMPPATH)/TESTV.dis
	
executeV:
	whisper -x  $(TEMPPATH)/programV.hex -s 0x80000000 --tohost 0xd0580000 -f  $(TEMPPATH)/logV.txt --configfile ./veer/whisper.json


allNV: compileNV executeNV

cleanNV: 
	rm -f $(TEMPPATH)/logNV.txt  $(TEMPPATH)/programNV.hex  $(TEMPPATH)/TESTNV.dis  $(TEMPPATH)/TESTNV.exe
	
compileNV:
	$(GCC_PREFIX)-gcc $(ABI) -lgcc -T$(LINK) -o  $(TEMPPATH)/TESTNV.exe $(CODEFOLDER)/FFT_NV.s -nostartfiles -lm
	$(GCC_PREFIX)-objcopy -O verilog  $(TEMPPATH)/TESTNV.exe  $(TEMPPATH)/programNV.hex
	$(GCC_PREFIX)-objdump -S  $(TEMPPATH)/TESTNV.exe >  $(TEMPPATH)/TESTNV.dis
	
executeNV:
	whisper -x  $(TEMPPATH)/programNV.hex -s 0x80000000 --tohost 0xd0580000 -f  $(TEMPPATH)/logNV.txt --configfile ./veer/whisper.json


allV2: compileV2 executeV2

cleanV2: 
	rm -f $(TEMPPATH)/logV2.txt  $(TEMPPATH)/programV2.hex  $(TEMPPATH)/TESTV2.dis  $(TEMPPATH)/TESTV2.exe
	
compileV2:
	$(GCC_PREFIX)-gcc $(ABI) -lgcc -T$(LINK) -o  $(TEMPPATH)/TESTV2.exe $(CODEFOLDER)/FFT_V2.s -nostartfiles -lm
	$(GCC_PREFIX)-objcopy -O verilog  $(TEMPPATH)/TESTV2.exe  $(TEMPPATH)/programV2.hex
	$(GCC_PREFIX)-objdump -S  $(TEMPPATH)/TESTV2.exe >  $(TEMPPATH)/TESTV2.dis
	
	
executeV2:
	whisper -x  $(TEMPPATH)/programV2.hex -s 0x80000000 --tohost 0xd0580000 -f  $(TEMPPATH)/logV2.txt --configfile ./veer/whisper.json


allNV2: compileNV2 executeNV2

cleanNV2: 
	rm -f $(TEMPPATH)/logNV2.txt  $(TEMPPATH)/programNV2.hex  $(TEMPPATH)/TESTNV2.dis  $(TEMPPATH)/TESTNV2.exe
	
compileNV2:
	$(GCC_PREFIX)-gcc $(ABI) -lgcc -T$(LINK) -o  $(TEMPPATH)/TESTNV2.exe $(CODEFOLDER)/FFT_NV2.s -nostartfiles -lm
	$(GCC_PREFIX)-objcopy -O verilog  $(TEMPPATH)/TESTNV2.exe  $(TEMPPATH)/programNV2.hex
	$(GCC_PREFIX)-objdump -S  $(TEMPPATH)/TESTNV2.exe >  $(TEMPPATH)/TESTNV2.dis
	
executeNV2:
	whisper -x  $(TEMPPATH)/programNV2.hex -s 0x80000000 --tohost 0xd0580000 -f  $(TEMPPATH)/logNV2.txt --configfile ./veer/whisper.json

alll: compilel executel

cleanl: 
	rm -f $(TEMPPATH)/logNV2.txt  $(TEMPPATH)/programNV2.hex  $(TEMPPATH)/TESTNV2.dis  $(TEMPPATH)/TESTNV2.exe
	
compilel:
	$(GCC_PREFIX)-gcc $(ABI) -lgcc -T$(LINK) -o  $(TEMPPATH)/TESTNV2.exe $(CODEFOLDER)/l.s -nostartfiles -lm
	$(GCC_PREFIX)-objcopy -O verilog  $(TEMPPATH)/TESTNV2.exe  $(TEMPPATH)/programNV2.hex
	$(GCC_PREFIX)-objdump -S  $(TEMPPATH)/TESTNV2.exe >  $(TEMPPATH)/TESTNV2.dis
	
executel:
	whisper -x  $(TEMPPATH)/programNV2.hex -s 0x80000000 --tohost 0xd0580000 -f  $(TEMPPATH)/logNV2.txt --configfile ./veer/whisper.json
