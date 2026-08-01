import type { FullConfig } from '@nekosu/maa-tools'

const config: FullConfig = {
  cwd: import.meta.dirname,
  maaVersion: 'latest',
  interfacePath: 'assets/interface.json',
  check: {
    override: {
      // 忽略 MPE（MaaPiEditor）编辑器元数据产生的警告，MaaFramework 运行时支持该格式
      'mpe-config': 'ignore'
    }
  }
}

export default config
