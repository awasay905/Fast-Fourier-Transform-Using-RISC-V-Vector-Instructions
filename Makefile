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
