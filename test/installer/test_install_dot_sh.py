import asyncio
from pathlib import Path


async def test_install_sh_shellcheck():
    install_sh = Path(__file__).parent.parent.parent / "install.sh"
    assert install_sh.exists(), "install.sh not found in repository root"

    process = await asyncio.create_subprocess_exec(
        "shellcheck",
        "--severity=error",
        str(install_sh),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    assert process.returncode == 0, f"shellcheck found errors:\n{stdout.decode()}\n{stderr.decode()}"
